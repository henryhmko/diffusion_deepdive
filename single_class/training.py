from single_class import *
from utils import *
import os
from tqdm import tqdm
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    unet = UNet(labels=False).to(device)
    diffusion_model = DiffusionModel()

    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")

        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = torch.randint(0, diffusion_model.timesteps, (images.shape[0],)).long().to(device)

            batch_noisy, noise = diffusion_model.forward(images, t, device)
            predicted_noise = unet(batch_noisy, t)

            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(noise, predicted_noise)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch*args.epochs + i)
        
        #sampling at the end of each epoch
        if epoch % args.sampling_frequency == 0:
            logging.info(f"Sampling 8 new images...")
            unet.eval()
            with torch.no_grad():
                #generate results with each diffusion step
                for img_num in tqdm(range(2), desc=f"Generating Diffusion Step Samples for Epoch: {epoch}..."): #generate 2 plots of diffusion process
                    plt.figure(figsize=(32, 32))
                    f, ax = plt.subplots(1, int(diffusion_model.timesteps / 50), figsize=(32*int(diffusion_model.timesteps / 50), 32))

                    idx = 0

                    img = torch.randn((1,3) + (args.image_size, args.image_size)).to(args.device)
                    for i in reversed(range(diffusion_model.timesteps)):
                        t = torch.full((1,), i, dtype=torch.long, device=args.device)
                        img = diffusion_model.reverse_process(img, t, unet.eval())
                        if i % 50 == 0:
                            ax[idx].imshow(reverse_transform(img[0]))
                            idx += 1

                    new_path = os.path.join("results", args.run_name, f"epoch_{epoch}_full_diffusion_{img_num}.png")
                    plt.tight_layout(pad=0)
                    plt.savefig(new_path)
                    plt.close('all')
                #generate 6 new samples showing only the results
                # generate summary of n images
                plt.figure(figsize=(32, 32))
                f, ax = plt.subplots(1, args.num_samples, figsize=(32*args.num_samples, 32))

                idx = 0
                for _ in tqdm(range(args.num_samples), desc="Generating Summary Plot..."):
                    img = torch.randn((1,3) + (args.image_size, args.image_size)).to(args.device)
                    for i in reversed(range(diffusion_model.timesteps)):
                        t = torch.full((1,), i, dtype=torch.long, device=args.device)
                        img = diffusion_model.reverse_process(img, t, unet.eval())
                    ax[idx].imshow(reverse_transform(img[0]))
                    idx += 1
                plt.tight_layout()
                new_path = os.path.join("results", args.run_name, f"epoch_{epoch}_summary_result.png")
                plt.savefig(new_path)
                plt.close('all')
        
            unet.train()

        #save model checkpoint
        torch.save(unet.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))    

def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Single_Class_Detailed_Sampling_lr=-4"
    args.epochs = 500
    args.sampling_frequency = 10 #sample every n epochs
    args.lr = 1e-4
    args.batch_size = 16
    args.verbose = True
    args.image_size = 64
    args.dataset_path = "/home/ko.hyeonmok/diffusion/multi_class/data/landscape_imgs"
    args.device = "cuda"
    args.num_samples = 6
    train(args)

if __name__ == '__main__':
    launch()
