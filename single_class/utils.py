import os
import torchvision
from torch.utils.data import DataLoader


def get_data(args):
    transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((args.image_size, args.image_size)),
    torchvision.transforms.ToTensor(), #entries between 0 and 1
    torchvision.transforms.Lambda(lambda t: (t*2) - 1), #make entries from [0,1] to [-1,1]
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)