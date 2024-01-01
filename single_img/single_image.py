import torch 
import torchvision
import torchvision.transforms as transforms
import urllib
import PIL
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda')

# Fetch that oski image
def get_sample_image()->PIL.Image.Image:
    url = 'https://cdn.vox-cdn.com/thumbor/qDwEb-hLqEroyf_6_XW4BydAtkc=/1400x0/filters:no_upscale()/cdn.vox-cdn.com/uploads/chorus_asset/file/11816637/usa_today_10376139.jpg'
    filename = 'oski.jpg'
    urllib.request.urlretrieve(url, filename)
    return PIL.Image.open(filename)

IMAGE_SHAPE = (64, 64) # Change this to argparse

## TRANSFORMATIONS
#PIL -> PyTorch tensor
transform = transforms.Compose([
    transforms.Resize(IMAGE_SHAPE),
    transforms.ToTensor(), #entries between 0 and 1
    transforms.Lambda(lambda t: (t*2) - 1), #make entries from [0,1] to [-1,1]
])

#PyTorch tensor -> PIL
reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t+1) / 2), #make entries from [-1,1] to [0,1]
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), #reorder from [Channel, Height, Width] to [Height, Width, Channel]
    transforms.Lambda(lambda t: t * 255.), #scale entries to between [0., 255.] to fit pixel space values
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), #convert into uint np arr
    transforms.ToPILImage(), #convert to PIL image
])


# Diffusion model - forward and reverse
class DiffusionModel:
    def __init__(self, start_schedule=1e-4, end_schedule=2e-2, timesteps=1000):
        """
        if
            betas = [0.1, 0.2, 0.3, ...]
                        then,
            alphas = [0.9, 0.8, 0.7, ...]
                        and,
            alphas_cumprod = [0.9, 0.9*0.8, 0.9*0.8*0.7, ...]
        """
        self.start_schedule = start_schedule
        self.end_schedule = end_schedule
        self.timesteps = timesteps
        self.betas = torch.linspace(start_schedule, end_schedule, timesteps)
        self.alphas = 1 - self.betas #per equation above
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0) #cumulative product

    def forward(self, x_0, t, device):
        """
        x_0: (B, C, H, W)
        t: (B, )
        """
        noise = torch.randn_like(x_0) #isotropic Gaussian noise
        sqrt_alpha_cumprod_t = self.get_index_from_list(self.alphas_cumprod.sqrt(), t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x_0.shape) #ORIGINAL with YES .sqrt()
        # sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list((1. - self.alphas_cumprod), t, x_0.shape) #EDITED with NO .sqrt(): per original paper

        mean = sqrt_alpha_cumprod_t.to(device) * x_0.to(device)
        variance = sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)

        return mean + variance, noise.to(device)

    @torch.no_grad()
    def reverse_process(self, x, t, model, **kwargs):
        """
        Calls the model to predict the noise in the image and returns the denoised image
        Applies noise to this image, if we are not in the last step yet
        """
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(torch.sqrt(1. - self.alphas_cumprod), t, x.shape) #ORIGINAL with YES .sqrt()
        # sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list((1. - self.alphas_cumprod), t, x.shape) #EDITED with NO .sqrt(): per original paper
        sqrt_recip_alphas_t = self.get_index_from_list(torch.sqrt(1.0 / self.alphas), t, x.shape)

        mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t, **kwargs) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = betas_t

        if t == 0: #we are done denoising
            return mean #clean image
        else:
            noise = torch.randn_like(x) #sample another isotropic gaussian noise
            variance = torch.sqrt(posterior_variance_t) * noise
            return mean + variance

    @staticmethod
    def get_index_from_list(values, t, x_shape):
        batch_size = t.shape[0]
        """
        pick the values from vals
        according to the indices stored in `t`
        """
        result = values.gather(-1, t.cpu())
        """
        if
        x_shape = (5, 3, 64, 64)
            -> len(x_shape) = 4
            -> len(x_shape) - 1 = 3

        and thus we reshape `out` to dims
        (batch_size, 1, 1, 1)

        """
        return result.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


#set training image
pil_image = get_sample_image()
pil_image = PIL.Image.open('oski.jpg')

torch_image = transform(pil_image)
diffusion_model = DiffusionModel()

# Positional embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


# one block for UNET
class Block(nn.Module):
    def __init__(self, channels_in, channels_out, time_embedding_dims, labels, num_filters = 3, downsample=True):
        super().__init__()

        self.time_embedding_dims = time_embedding_dims
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dims)
        self.labels = labels
        if labels:
            self.label_mlp = nn.Linear(1, channels_out)

        self.downsample = downsample

        if downsample:
            self.conv1 = nn.Conv2d(channels_in, channels_out, num_filters, padding=1)
            self.final = nn.Conv2d(channels_out, channels_out, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(2 * channels_in, channels_out, num_filters, padding=1)
            self.final = nn.ConvTranspose2d(channels_out, channels_out, 4, 2, 1)

        self.bnorm1 = nn.BatchNorm2d(channels_out)
        self.bnorm2 = nn.BatchNorm2d(channels_out)

        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.time_mlp = nn.Linear(time_embedding_dims, channels_out)
        self.relu = nn.ReLU()

    def forward(self, x, t, **kwargs):
        o = self.bnorm1(self.relu(self.conv1(x)))
        o_time = self.relu(self.time_mlp(self.time_embedding(t)))
        o = o + o_time[(..., ) + (None, ) * 2]
        if self.labels:
            label = kwargs.get('labels')
            o_label = self.relu(self.label_mlp(label))
            o = o + o_label[(..., ) + (None, ) * 2]

        o = self.bnorm2(self.relu(self.conv2(o)))

        return self.final(o)
    

# Build UNet
class UNet(nn.Module):
    def __init__(self, img_channels = 3, time_embedding_dims = 128, labels = False, sequence_channels = (64, 128, 256, 512, 1024)):
        super().__init__()
        self.time_embedding_dims = time_embedding_dims
        sequence_channels_rev = reversed(sequence_channels)

        self.downsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels) for channels_in, channels_out in zip(sequence_channels, sequence_channels[1:])])
        self.upsampling = nn.ModuleList([Block(channels_in, channels_out, time_embedding_dims, labels,downsample=False) for channels_in, channels_out in zip(sequence_channels[::-1], sequence_channels[::-1][1:])])
        self.conv1 = nn.Conv2d(img_channels, sequence_channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(sequence_channels[0], img_channels, 1)


    def forward(self, x, t, **kwargs):
        residuals = []
        o = self.conv1(x)
        for ds in self.downsampling:
            o = ds(o, t, **kwargs)
            residuals.append(o)
        for us, res in zip(self.upsampling, reversed(residuals)):
            o = us(torch.cat((o, res), dim=1), t, **kwargs)

        return self.conv2(o)
    

