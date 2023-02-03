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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '../'))
from project.cartoonX import CartoonX
from project.pixelRDE import PixelRDE
import json
now = datetime.now()
current_time = now.strftime("%d/%m/%Y %H:%M:%S")

# Get list of imagenet labels to convert prediction to string label
LABEL_LIST = tuple(open(os.path.join(sys.path[0], "imagenet_labels.txt")).read().split('\n'))
LABEL_LIST = [x.replace('{', "").replace('\'', "").replace(',', "").replace('-', " ").replace('_', " ") for x in
              LABEL_LIST]


def main(imgdir, logdir, tensorboard, resize_images):
    # Get device (use GPU if possible)
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get classifier to explain
    model = models.mobilenet_v3_small(pretrained=True).eval().to(device)
    # Get hyperparameters for wavelet RDE and pixel RDE

    # Get files of images
    files = os.listdir(imgdir)

    # Explain model decsision for each image in files
    distortions_no_scaling = []
    distortions_scaling = []
    distortions_exp_scaling = []
    distortions_inverse_linear = []
    list_names = ['distortions_no_scaling', 'distortions_scaling', 'distortions_exp_scaling', 'distortions_inverse_linear']
    perturbations = ["Original Image", "CartoonX no Scaling", "CartoonX Linear Scaling","CartoonX Exponential Scaling", "Inverted Linear Scaling",
                     ]
    labels = ["CartoonX no Scaling", "CartoonX Linear Scaling","CartoonX Exponential Scaling", "Inverted Linear Scaling",
                     ]
    for fname in tqdm(files):
        print(f"Processing file: {fname}")
        # Get image and transform to tensor
        x = Image.open(os.path.join(imgdir, fname))
        x = transforms.ToTensor()(x)
        if resize_images: x = transforms.Resize(size=(256, 256))(x)
        x = x.to(device)

        # Get prediction for x
        output = model(x.unsqueeze(0).detach())
        max_idx = nn.Softmax(dim=1)(output).max(1)[1].item()
        label = LABEL_LIST[max_idx]

        fig, axs = plt.subplots(1, 5, figsize=(10, 10))
        masks = {
            'cartoon_mask': [],
            'pixel_mask': [],
        }
        scaling = ['normal', 'linear', 'exponential','inverse_linear']
        for idx, perturbation in enumerate(perturbations):
            with open(os.path.join(sys.path[0], f"hparams/obfuscation_experiment_hparams/hparams.yaml")) as f:
                HPARAMS_CARTOONX = yaml.load(f, Loader=yaml.FullLoader)["CartoonX"]
            if idx != 0:
                # Initialize wavelet RDE and pixel RDE
                cartoonX = CartoonX(model=model, device=device, **HPARAMS_CARTOONX)

                starttime_1 = time.time()
                img, cartoon_mask, cartoon_logs, _ = cartoonX(x.unsqueeze(0), target=max_idx, scaling = scaling[idx-1])
                print(f"\nTime: {time.time() - starttime_1}")
                masks['cartoon_mask'] = cartoon_mask
                print(f'distortion scaling with {scaling[idx-1]} wavelet : {cartoon_logs["distortion"][-1]}')
                vars()[list_names[idx-1]].append(cartoon_logs["distortion"][-1])

            else:
                img = x

            args = {}
            axs[idx].imshow(np.asarray(transforms.ToPILImage()(img)), vmin=0, vmax=255, **args)
            axs[idx].set_title(perturbation, size=8)
            axs[idx].axis("off")

        if tensorboard:
            # Log to tensorboard
            writer = SummaryWriter(os.path.join(logdir, f"image{fname}"))
            writer.add_figure(f"Explanations-{current_time}", fig)
            writer.flush()
            writer.close()
        else:
            if not os.path.isdir(logdir):
                os.makedirs(logdir)
            plt.savefig(os.path.join(logdir, f"exp-obf-scaling-{fname}"), bbox_inches='tight', transparent=True, pad_inches=0)

    print(f"distortions no scaling: {distortions_no_scaling}")
    print(f"distortions scaling: {distortions_scaling}")
    print(f"distortions exp scaling: {distortions_exp_scaling}")
    print(f'distortions removed lower: {distortions_inverse_linear}')
    print(f"mean distortion no scaling: {np.mean(distortions_no_scaling)}")
    print(f"mean distortion scaling: {np.mean(distortions_scaling)}")
    print(f"mean distortion exp scaling: {np.mean(distortions_exp_scaling)}")
    print(f"mean distortion removed lower: {np.mean(distortions_inverse_linear)}")
    print(f"std distortion no scaling: {np.std(distortions_no_scaling)}")
    print(f"std distortion scaling: {np.std(distortions_scaling)}")
    print(f"std distortion exp scaling: {np.std(distortions_exp_scaling)}")
    print(f"std distortion removed lower: {np.std(distortions_inverse_linear)}")

    x = [1,2,3,4]
    y = [np.mean(distortions_no_scaling), np.mean(distortions_scaling), np.mean(distortions_exp_scaling), np.mean(distortions_inverse_linear)]
    e = [np.std(distortions_no_scaling), np.std(distortions_scaling), np.std(distortions_exp_scaling), np.std(distortions_inverse_linear)]
    plt.errorbar(x,y,e,linestyle='None', marker='o')
    plt.xticks(x, labels)
    plt.ylabel('Distortion')
    plt.title('Distortion of CartoonX Scaling Experiment')
    plt.show()

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgdir", type=str, help="Directory of images to explain.", default=".")
    parser.add_argument("--logdir", type=str, help="Directory where explanations are logged", default="exp_logs")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    parser.add_argument("--resize_images", dest="resize_images", action="store_true")
    args = parser.parse_args()
    main(imgdir=args.imgdir, logdir=args.logdir, resize_images=args.resize_images, tensorboard=args.tensorboard)
