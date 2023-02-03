import argparse
import os
import sys
import yaml
from datetime import datetime

import time

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from plotting_functions import figure_8a_plottinng, figure_8b_plotting, figure_8c_plotting, figure_8d_plotting, \
    figure_6_plotting

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from project.cartoonX import CartoonX
from project.pixelRDE import PixelRDE

# Get current time for logging
now = datetime.now()
current_time = now.strftime("%d/%m/%Y %H:%M:%S")

# Get list of imagenet labels to convert prediction to string label
LABEL_LIST = tuple(open(os.path.join(sys.path[0], "imagenet_labels.txt")).read().split('\n'))
LABEL_LIST = [x.replace('{', "").replace('\'', "").replace(',', "").replace('-', " ").replace('_', " ") for x in
              LABEL_LIST]


def main(imgdir, logdir, tensorboard, resize_images, figure_name="8a"):
    # Get device (use GPU if possible)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model Selection

    if figure_name == "5b":
        model = models.vgg16(pretrained=True).eval().to(device)

    else:
        model = models.mobilenet_v3_small(pretrained=True).eval().to(device)


    # Get files of images
    files = os.listdir(imgdir)

    # Explain model decsision for each image in files
    if figure_name == "8a":
        set_of_images = [f'Original Image', '$\lambda$ = 0', '$\lambda$ = 50', '$\lambda$ = 100', '$\lambda$ = 200',
                        '$\lambda$ = 400', '$\lambda$ = 800']

        hparam_folder = "hparams/figure8a"
        plot_x_dim = 7
        plot_y_dim = 2
        saved_figure_name = "fig8a"

    if figure_name == "8b":
        set_of_images = ["Original Image", "Gaussian", "Zero Baseline"]
        hparam_folder = "hparams/figure8b"
        plot_x_dim = 3
        plot_y_dim = 1
        saved_figure_name = "fig8b"

    if figure_name == "8c":
        set_of_images = ['Original Image', 'db5', 'db4', 'db3', 'Haar Wavelet']
        hparam_folder = "hparams/figure8c"
        plot_x_dim = 5
        plot_y_dim = 1
        saved_figure_name = "fig8c"

    if figure_name == "8d":
        set_of_images = ["Original Image", "Squared $l_2$ in Label", "Maximize Label", "$l_2$ Probabilities", "KL-Divergence"]
        hparam_folder = "hparams/figure8d"
        plot_x_dim = 5
        plot_y_dim = 1
        saved_figure_name = "fig8d"
        target = None

    if figure_name[0] == "5":
        set_of_images = ["Original Image", "CartoonX", "PixelRDE"]
        hparam_folder = "hparams/figure6"
        plot_x_dim = 3
        plot_y_dim = 1
        if figure_name[-1] == 'a':
            saved_figure_name = "fig5a"
        if figure_name[-1] == 'b':
            saved_figure_name = "fig5b"
    if figure_name[0] == "6":
        set_of_images = ["Original Image", "CartoonX", "PixelRDE"]
        hparam_folder = "hparams/figure6"
        plot_x_dim = 3
        plot_y_dim = 1
        if figure_name[-1] == 'a':
            saved_figure_name = "fig6a"
        if figure_name[-1] == 'b':
            saved_figure_name = "fig6b"


    for fname in tqdm(files):
        print(f"Processing file: {fname}")
        # Get image and transform to tensor
        x = Image.open(os.path.join(imgdir, fname))
        x = transforms.ToTensor()(x)
        if resize_images: x = transforms.Resize(size=(256, 256))(x)
        x = x.to(device)

        output = model(x.unsqueeze(0).detach())
        probs = nn.Softmax(dim=1)(output)
        max_prob = torch.max(probs).item()
        max_idx = nn.Softmax(dim=1)(output).max(1)[1].item()
        label = LABEL_LIST[max_idx]

        # Figure 8a Recreation

        fig, axs = plt.subplots(plot_y_dim, plot_x_dim, constrained_layout = True)

        for idx, wavelet in enumerate(set_of_images):
            if idx > 0:
                with open(os.path.join(sys.path[0], f"{hparam_folder}/hparams_{idx}.yaml")) as f:
                    HPARAMS_CARTOONX = yaml.load(f, Loader=yaml.FullLoader)["CartoonX"]
                starttime_1 = time.time()
                cartoonX = CartoonX(model=model, device=device, **HPARAMS_CARTOONX)

                if figure_name == "8d":
                    if idx == 2 or idx == 4:
                        img, cartoon_mask, cartoon_logs, _ = cartoonX(x.unsqueeze(0), target=None, retain_graph=True)
                    elif idx == 3:
                        img, cartoon_mask, cartoon_logs, _ = cartoonX(x.unsqueeze(0), target=[probs, 1],retain_graph=True)
                    else:
                        img, cartoon_mask, cartoon_logs, _ = cartoonX(x.unsqueeze(0), target=max_idx, retain_graph=True)

                else:
                    img_CartoonX, cartoon_mask, cartoon_logs, _ = cartoonX(x.unsqueeze(0), target=max_idx)
                    print(f"\n{time.time() - starttime_1}")
                    if figure_name == "8a" or figure_name[0] == "6" or figure_name[0] == "5":
                        with open(os.path.join(sys.path[0], f"{hparam_folder}/hparams_{idx}.yaml")) as f:
                            HPARAMS_PIXEL_RDE = yaml.load(f, Loader=yaml.FullLoader)["PixelRDE"]

                        pixelRDE = PixelRDE(model=model, device=device, **HPARAMS_PIXEL_RDE)
                        starttime_2 = time.time()
                        img_pixelRDE, pixelrde_mask, pixelrde_logs = pixelRDE(x.unsqueeze(0), target=max_idx)
                        print(f"\n{time.time() - starttime_2}")
            else:
                img_CartoonX = x
                if figure_name == "8a" or figure_name[0] == "6" or figure_name[0] == "5":
                    img_pixelRDE = x
            if figure_name == "8a":
                figure_8a_plottinng(idx, axs, img_CartoonX, img_pixelRDE, wavelet)
            if figure_name == "8b":
                figure_8b_plotting(idx, axs, img_CartoonX, wavelet)
            if figure_name == "8c":
                figure_8c_plotting(idx, axs, img_CartoonX, wavelet)
            if figure_name == "8d":
                figure_8d_plotting(idx, axs, img_CartoonX, wavelet)

            if figure_name[0] == "6" or figure_name[0] == "5":
                figure_6_plotting(idx, axs, img_CartoonX, wavelet)

        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        plt.savefig(os.path.join(logdir, f"exp-{saved_figure_name}-{fname}"), bbox_inches='tight', transparent=True, pad_inches=0)

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgdir", type=str, help="Directory of images to explain.", default=".")
    parser.add_argument("--logdir", type=str, help="Directory where explanations are logged", default="exp_logs")
    parser.add_argument("--tensorboard", dest="tensorboard", action="store_true")
    parser.add_argument("--resize_images", dest="resize_images", action="store_true")
    parser.add_argument("--figure_name",type=str, help="Name of figure to recreate", default="8a")
    args = parser.parse_args()
    main(imgdir=args.imgdir, logdir=args.logdir, resize_images=args.resize_images, tensorboard=args.tensorboard, figure_name=args.figure_name)