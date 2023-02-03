import numpy as np
from torchvision import transforms


def figure_8a_plottinng(idx, axs,img_CartoonX,img_pixelRDE, wavelet):
    args = {"cmap": "gray"} if idx > 0 else {}
    axs[0, idx].imshow(np.asarray(transforms.ToPILImage()(img_CartoonX)), vmin=0, vmax=255, **args)
    axs[0, idx].set_title(wavelet, size=8)
    axs[0, idx].axis("off")

    args = {"cmap": "copper"} if idx > 0 else {}
    axs[1, idx].imshow(np.asarray(transforms.ToPILImage()(img_pixelRDE)), vmin=0, vmax=255, **args)
    axs[1, idx].set_title(wavelet, size=8)
    axs[1, idx].axis("off")

def figure_8b_plotting(idx, axs, img_CartoonX, wavelet):
    args = {"cmap": "gray"} if idx > 0 else {}
    axs[idx].imshow(np.asarray(transforms.ToPILImage()(img_CartoonX)), vmin=0, vmax=255, **args)
    axs[idx].set_title(wavelet, size=8)
    axs[idx].axis("off")

def figure_8c_plotting(idx, axs, img_CartoonX, wavelet):
    args = {"cmap": "gray"} if idx > 0 else {}
    axs[idx].imshow(np.asarray(transforms.ToPILImage()(img_CartoonX)), vmin=0, vmax=255, **args)
    axs[idx].set_title(wavelet, size=8)
    axs[idx].axis("off")

def figure_8d_plotting(idx, axs, img_CartoonX, wavelet):
    args = {"cmap": "gray"} if idx > 0 else {}
    axs[idx].imshow(np.asarray(transforms.ToPILImage()(img_CartoonX)), vmin=0, vmax=255, **args)
    axs[idx].set_title(wavelet, size=8)
    axs[idx].axis("off")

def figure_6_plotting(idx, axs, img_CartoonX, wavelet):
    if idx == 1:
        args = {'cmap': 'gray'}
    elif idx == 2:
        args = {'cmap': 'copper'}
    else:
        args = {}
    axs[idx].imshow(np.asarray(transforms.ToPILImage()(img_CartoonX)), vmin=0, vmax=255, **args)
    axs[idx].set_title(wavelet, size=8)
    axs[idx].axis("off")