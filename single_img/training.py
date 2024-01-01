from single_image import *
import os
from tqdm import tqdm, trange

# get this to argparse
NO_EPOCHS = 2001
PRINT_FREQUENCY = 400
LR = 1e-3
BATCH_SIZE = 16
VERBOSE = True

#empty gpu memory 
torch.cuda.empty_cache()

device = torch.device('cuda')
weights_folder = '/home/ko.hyeonmok/diffusion/single_img/weights/'

# initialize unet
unet = UNet(labels=False)
# diffusion_model = DiffusionModel()

unet.to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

for epoch in trange(0, NO_EPOCHS, desc=f'[Full Loop]', leave=True):
    mean_epoch_loss = []

    batch = torch.stack([torch_image] * BATCH_SIZE)
    t = torch.randint(0, diffusion_model.timesteps, (BATCH_SIZE,)).long().to(device)

    batch_noisy, noise = diffusion_model.forward(batch, t, device)
    predicted_noise = unet(batch_noisy, t)

    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(noise, predicted_noise)
    mean_epoch_loss.append(loss.item())
    loss.backward()
    optimizer.step()

    if epoch % PRINT_FREQUENCY == 0:
        print('---')
        print(f"Epoch: {epoch} | Train Loss {np.mean(mean_epoch_loss)}")

    if epoch == NO_EPOCHS - 1: #last epoch
        state = {
            'epoch': epoch,
            'state_dict': unet.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        if not os.path.exists(weights_folder):
            os.mkdir(weights_folder)

        torch.save(unet, weights_folder+f'epoch_{epoch}_MODEL.ckpt')

with torch.no_grad():
  #make plot
  plt.figure(figsize=(10, 10))
  f, ax = plt.subplots(1, int(diffusion_model.timesteps / 50), figsize=(100, 100))

  idx = 0

  img = torch.randn((1,3) + IMAGE_SHAPE).to(device)
  for i in reversed(range(diffusion_model.timesteps)):
    t = torch.full((1,), i, dtype=torch.long, device=device)
    img = diffusion_model.reverse_process(img, t, unet.eval())
    if i % 50 == 0:
      ax[idx].imshow(reverse_transform(img[0]))
      idx += 1
#   plt.show()
  plt.savefig('transform_img.png')
