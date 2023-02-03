import torch

HAVE_GPU = torch.cuda.is_available()
if HAVE_GPU:
    dev = torch.device('cuda')
    print(dev)
else:
    dev = torch.device('cpu')

