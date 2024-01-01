import torch
from single_class import *

import argparse
import tqdm
import matplotlib.pyplot as plt
import os
from tqdm import tqdm



def sample(args, n):
    unet = UNet()
    unet.load_state_dict(torch.load(args.model_path))
    unet.eval().to(args.device)

    diffusion_model = DiffusionModel()

    with torch.no_grad():
        #generate results with each diffusion step
        for img_num in tqdm(range(5), desc="Generating Diffusion Step Samples..."):
            plt.figure(figsize=(10, 10))
            f, ax = plt.subplots(1, int(diffusion_model.timesteps / 50), figsize=(100, 100))

            idx = 0

            img = torch.randn((1,3) + (args.img_size, args.img_size)).to(args.device)
            for i in reversed(range(diffusion_model.timesteps)):
                t = torch.full((1,), i, dtype=torch.long, device=device)
                img = diffusion_model.reverse_process(img, t, unet.eval())
                if i % 50 == 0:
                    ax[idx].imshow(reverse_transform(img[0]))
                    idx += 1
            #   plt.show()
            new_path = os.path.join("results", args.inference_name, f"result_img_{img_num}.png")
            plt.savefig(new_path)
            plt.close('all')

        # generate summary of n images
        plt.figure(figsize=(32, 32))
        f, ax = plt.subplots(1, n, figsize=(32*n, 32))

        idx = 0
        for _ in tqdm(range(n), desc="Generating Summary Plot..."):
            img = torch.randn((1,3) + (args.img_size, args.img_size)).to(args.device)
            for i in reversed(range(diffusion_model.timesteps)):
                t = torch.full((1,), i, dtype=torch.long, device=device)
                img = diffusion_model.reverse_process(img, t, unet.eval())
            ax[idx].imshow(reverse_transform(img[0]))
            idx += 1
        plt.tight_layout()
        new_path = os.path.join("results", args.inference_name, f"summary_result_img.png")
        plt.savefig(new_path)
        plt.close('all')

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.model_path = "/home/ko.hyeonmok/diffusion/single_class/models/Single_Class_lr=-5/ckpt.pt"
    args.inference_name = "Single_Class_lr=-5"
    args.device = 'cuda'
    args.img_size = 64
    args.timesteps = 1000

    sample(args, 6)


if __name__ == '__main__':
    launch()