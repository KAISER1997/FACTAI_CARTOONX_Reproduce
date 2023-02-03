import argparse
import os
import sys
import yaml
from shutil import copyfile
from datetime import datetime

import time

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from project.cartoonX import CartoonX
from project.pixelRDE import PixelRDE

from pytorch_wavelets import DWTForward, DWTInverse

# Get current time for logging
now = datetime.now()
current_time = now.strftime("%d/%m/%Y %H:%M:%S")

# Get list of imagenet labels to convert prediction to string label
LABEL_LIST = tuple(open(os.path.join(sys.path[0], "imagenet_labels.txt")).read().split('\n'))
LABEL_LIST = [x.replace('{', "").replace('\'', "").replace(',', "").replace('-', " ").replace('_', " ") for x in
              LABEL_LIST]


def main(imgdir, logdir, tensorboard, resize_images):
    # Get device (use GPU if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get classifier to explain
    model = models.mobilenet_v3_small(pretrained=True).eval().to(device)

    # Get files of images
    files = os.listdir(imgdir)

    pixel_rde_distortions = []
    pixel_rde_l1_norms = []
    cartoonx_distortions = []
    cartoonx_l1_norms = []
    lambda_vals = [4,8,16,32,64]

    i=0

    for experiment in os.listdir(os.path.join(sys.path[0], 'hparams/figure7c_experiments')):
        for fname in tqdm(files):
            print(f"Processing file: {fname}")
            # Get image and transform to tensor
            x = Image.open(os.path.join(imgdir, fname))
            x = transforms.ToTensor()(x)
            if resize_images: x = transforms.Resize(size=(256, 256))(x)
            x = x.to(device)

            # Get prediction for x
            output = model(x.unsqueeze(0).detach())
            probs = nn.Softmax(dim=1)(output)
            max_prob = torch.max(probs).item()
            max_idx = nn.Softmax(dim=1)(output).max(1)[1].item()
            label = LABEL_LIST[max_idx]

            img_pixel_rde_distortions = []
            img_pixel_rde_l1_norms = []
            img_cartoonx_distortions = []
            img_cartoonx_l1_norms = []
            for idx in range(1,6):
                with open(os.path.join(sys.path[0], f"hparams/figure7c_experiments/{experiment}/hparams_{idx}.yaml")) as f:
                    HPARAMS_CARTOONX = yaml.load(f, Loader=yaml.FullLoader)["CartoonX"]
                    cartoonX = CartoonX(model=model, device=device, **HPARAMS_CARTOONX)
                with open(os.path.join(sys.path[0], f"hparams/figure7c_experiments/{experiment}/hparams_{idx}.yaml")) as f:
                    img, cartoon_mask, cartoon_logs, dists = cartoonX(x.unsqueeze(0), target=max_idx)
                    HPARAMS_PIXEL_RDE = yaml.load(f, Loader=yaml.FullLoader)["PixelRDE"]

                    pixelRDE = PixelRDE(model=model, device=device, **HPARAMS_PIXEL_RDE)
                    exp_pixelRDE, pixelrde_mask, pixelrde_logs = pixelRDE(x.unsqueeze(0), target=max_idx)

                img_pixel_rde_distortions.append(min(pixelrde_logs["distortion"]))
                img_pixel_rde_l1_norms.append(min(pixelrde_logs["l1-norm"]))
                img_cartoonx_distortions.append(min(cartoon_logs["distortion"]))
                img_cartoonx_l1_norms.append(min(cartoon_logs["l1-norm"]))
            pixel_rde_distortions.append(img_pixel_rde_distortions)
            pixel_rde_l1_norms.append(img_pixel_rde_l1_norms)
            cartoonx_distortions.append(img_cartoonx_distortions)
            cartoonx_l1_norms.append(img_cartoonx_l1_norms)


            if not os.path.isdir(logdir):
                os.makedirs(logdir)

        print(f'Pixel RDE Distortion: {pixel_rde_distortions}')
        print(f'Pixel RDE L1 Norm: {pixel_rde_l1_norms}')
        print(f'CartoonX Distortion: {cartoonx_distortions}')
        print(f'CartoonX L1 Norm: {cartoonx_l1_norms}')

        #Normalize the data and plot it against the lambda values
        pixel_rde_distortion_total = np.mean(np.array(pixel_rde_distortions), axis=0)/np.linalg.norm(np.mean(np.array(pixel_rde_distortions), axis=0))
        pixel_rde_l1_norms_total = np.mean(np.array(pixel_rde_l1_norms), axis=0)/np.linalg.norm(np.mean(np.array(pixel_rde_l1_norms), axis=0))
        cartoonx_distortions_total = np.mean(np.array(cartoonx_distortions), axis=0)/np.linalg.norm(np.mean(np.array(cartoonx_distortions), axis=0))
        cartoonx_l1_norms_total = np.mean(np.array(cartoonx_l1_norms), axis=0)/np.linalg.norm(np.mean(np.array(cartoonx_l1_norms), axis=0))
        x_ticks_names = ['$2^2$', '$2^3$', '$2^4$', '$2^5$', '$2^6$']

        plt.xscale('log')
        plt.xticks(lambda_vals, x_ticks_names)
        plt.plot(lambda_vals, pixel_rde_distortion_total, '-o', label='Pixel RDE Distortion', color='red')
        plt.plot(lambda_vals, pixel_rde_l1_norms_total, '--o', label='Pixel RDE L1 Norm $\\frac{||s||_1}{k}$', color='red')
        plt.plot(lambda_vals, cartoonx_distortions_total,'-o', label='CartoonX Distortion', color='blue')
        plt.plot(lambda_vals, cartoonx_l1_norms_total,'--o', label='CartoonX L1 Norm $\\frac{||s||_1}{k}$', color='blue')
        plt.grid()
        plt.legend()
        plt.xlabel('$\\lambda k$')

        plt.savefig(os.path.join(logdir, f"fig7c_{i}"), bbox_inches='tight', transparent=False, pad_inches=0)
        plt.show()
        i+=1

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgdir", type=str, help="Directory of images to explain.", default=".")
    parser.add_argument("--logdir", type=str, help="Directory where explanations are logged", default="exp_logs")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    parser.add_argument("--resize_images", dest="resize_images", action="store_true")
    args = parser.parse_args()
    main(imgdir=args.imgdir, logdir=args.logdir, resize_images=args.resize_images, tensorboard=args.tensorboard)
