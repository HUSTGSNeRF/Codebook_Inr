import numpy as np
import torch
def psnr(img1, img2):
    """Calculates PSNR between two images.
    img* have range [0,1]
    Args:
        img1 (torch.Tensor):
        img2 (torch.Tensor):
    """
    mse = torch.mean((img1-img2)**2)
    return 20. * torch.log10(1.0/torch.sqrt(mse)).to('cpu').item()


def clamp_image(img):
    """Clamp image values from [-1,1] to like in [0, 1].
    Args:
        img (torch.Tensor):
    return: img_ in [0,1]
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, -1., 1.) # 
    img_ =  img_ * 0.5 + 0.5
    return img_

def clamp_255(img):
    """Clamp image values from [-1,1] to like in [0, 1].
    Args:
        img (torch.Tensor):
    return: img_ in [0,1]
    """
    # Values may lie outside [0, 1], so clamp input
    img_ = torch.clamp(img, -1., 1.) # 
    img_ =  img_ * 0.5 + 0.5
    img_ = img_ * 255
    return img_