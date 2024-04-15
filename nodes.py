# import os
# import sys
import math
import copy
import torch
# import torchvision.transforms
import numpy as np
import cv2
# from pymatting import *
from pymatting import estimate_alpha_cf, estimate_foreground_ml, fix_trimap
from tqdm import trange

try:
    from cv2.ximgproc import guidedFilter
except ImportError:
    print("\033[33mUnable to import guidedFilter, make sure you have only opencv-contrib-python or run the import_error_install.bat script\033[m")

import comfy.model_management
from comfy.utils import ProgressBar
from comfy_extras.nodes_post_processing import gaussian_kernel
from .raft import *

MAX_RESOLUTION=8192

class AlphaClean:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "fill_holes": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 16,
                    "step": 1
                }),
                "white_threshold": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
                "extra_clip": ("FLOAT", {
                    "default": 0.98,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "alpha_clean"

    CATEGORY = "image/filters"

    def alpha_clean(self, images: torch.Tensor, radius: int, fill_holes: int, white_threshold: float, extra_clip: float):
        
        d = radius * 2 + 1
        i_dup = copy.deepcopy(images.cpu().numpy())
        
        for index, image in enumerate(i_dup):
            
            cleaned = cv2.bilateralFilter(image, 9, 0.05, 8)
            
            alpha = np.clip((image - white_threshold) / (1 - white_threshold), 0, 1)
            rgb = image * alpha
            
            alpha = cv2.GaussianBlur(alpha, (d,d), 0) * 0.99 + np.average(alpha) * 0.01
            rgb = cv2.GaussianBlur(rgb, (d,d), 0) * 0.99 + np.average(rgb) * 0.01
            
            rgb = rgb / np.clip(alpha, 0.00001, 1)
            rgb = rgb * extra_clip
            
            cleaned = np.clip(cleaned / rgb, 0, 1)
            
            if fill_holes > 0:
                fD = fill_holes * 2 + 1
                gamma = cleaned * cleaned
                kD = np.ones((fD, fD), np.uint8)
                kE = np.ones((fD + 2, fD + 2), np.uint8)
                gamma = cv2.dilate(gamma, kD, iterations=1)
                gamma = cv2.erode(gamma, kE, iterations=1)
                gamma = cv2.GaussianBlur(gamma, (fD, fD), 0)
                cleaned = np.maximum(cleaned, gamma)

            i_dup[index] = cleaned
        
        return (torch.from_numpy(i_dup),)

class AlphaMatte:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha_trimap": ("IMAGE",),
                "preblur": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 256,
                    "step": 1
                }),
                "blackpoint": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 0.99,
                    "step": 0.01
                }),
                "whitepoint": ("FLOAT", {
                    "default": 0.99,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
                "max_iterations": ("INT", {
                    "default": 1000,
                    "min": 100,
                    "max": 10000,
                    "step": 100
                }),
                "estimate_fg": (["true", "false"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE",)
    RETURN_NAMES = ("alpha", "fg", "bg",)
    FUNCTION = "alpha_matte"

    CATEGORY = "image/filters"

    def alpha_matte(self, images, alpha_trimap, preblur, blackpoint, whitepoint, max_iterations, estimate_fg):
        
        d = preblur * 2 + 1
        
        i_dup = copy.deepcopy(images.cpu().numpy().astype(np.float64))
        a_dup = copy.deepcopy(alpha_trimap.cpu().numpy().astype(np.float64))
        fg = copy.deepcopy(images.cpu().numpy().astype(np.float64))
        bg = copy.deepcopy(images.cpu().numpy().astype(np.float64))
        
        
        for index, image in enumerate(i_dup):
            trimap = a_dup[index][:,:,0] # convert to single channel
            if preblur > 0:
                trimap = cv2.GaussianBlur(trimap, (d, d), 0)
            trimap = fix_trimap(trimap, blackpoint, whitepoint)
            
            alpha = estimate_alpha_cf(image, trimap, laplacian_kwargs={"epsilon": 1e-6}, cg_kwargs={"maxiter":max_iterations})
            
            if estimate_fg == "true":
                fg[index], bg[index] = estimate_foreground_ml(image, alpha, return_background=True)
            
            a_dup[index] = np.stack([alpha, alpha, alpha], axis = -1) # convert back to rgb
        
        return (
            torch.from_numpy(a_dup.astype(np.float32)), # alpha
            torch.from_numpy(fg.astype(np.float32)), # fg
            torch.from_numpy(bg.astype(np.float32)), # bg
            )

def RGB2YCbCr(t):
    YCbCr = t.detach().clone()
    YCbCr[:,:,:,0] = 0.2123 * t[:,:,:,0] + 0.7152 * t[:,:,:,1] + 0.0722 * t[:,:,:,2]
    YCbCr[:,:,:,1] = 0 - 0.1146 * t[:,:,:,0] - 0.3854 * t[:,:,:,1] + 0.5 * t[:,:,:,2]
    YCbCr[:,:,:,2] = 0.5 * t[:,:,:,0] - 0.4542 * t[:,:,:,1] - 0.0458 * t[:,:,:,2]
    return YCbCr

def YCbCr2RGB(t):
    RGB = t.detach().clone()
    RGB[:,:,:,0] = t[:,:,:,0] + 1.5748 * t[:,:,:,2]
    RGB[:,:,:,1] = t[:,:,:,0] - 0.1873 * t[:,:,:,1] - 0.4681 * t[:,:,:,2]
    RGB[:,:,:,2] = t[:,:,:,0] + 1.8556 * t[:,:,:,1]
    return RGB

class BetterFilmGrain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 2.0, "step": 0.05}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "toe": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.5, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "grain"

    CATEGORY = "image/filters"

    def grain(self, image, scale, strength, saturation, toe, seed):
        t = image.detach().clone()
        torch.manual_seed(seed)
        grain = torch.rand(t.shape[0], int(t.shape[1] // scale), int(t.shape[2] // scale), 3)
        
        YCbCr = RGB2YCbCr(grain)
        YCbCr[:,:,:,0] = cv_blur_tensor(YCbCr[:,:,:,0], 3, 3)
        YCbCr[:,:,:,1] = cv_blur_tensor(YCbCr[:,:,:,1], 15, 15)
        YCbCr[:,:,:,2] = cv_blur_tensor(YCbCr[:,:,:,2], 11, 11)
        
        grain = (YCbCr2RGB(YCbCr) - 0.5) * strength
        grain[:,:,:,0] *= 2
        grain[:,:,:,2] *= 3
        grain += 1
        grain = grain * saturation + grain[:,:,:,1].unsqueeze(3).repeat(1,1,1,3) * (1 - saturation)
        
        grain = torch.nn.functional.interpolate(grain.movedim(-1,1), size=(t.shape[1], t.shape[2]), mode='bilinear').movedim(1,-1)
        t[:,:,:,:3] = torch.clip((1 - (1 - t[:,:,:,:3]) * grain) * (1 - toe) + toe, 0, 1)
        return(t,)

class BlurImageFast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "radius_x": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1023,
                    "step": 1
                }),
                "radius_y": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1023,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blur_image"

    CATEGORY = "image/filters"

    def blur_image(self, images, radius_x, radius_y):
        
        if radius_x + radius_y == 0:
            return (images,)
        
        dx = radius_x * 2 + 1
        dy = radius_y * 2 + 1
        
        dup = copy.deepcopy(images.cpu().numpy())
        
        for index, image in enumerate(dup):
            dup[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        
        return (torch.from_numpy(dup),)

class BlurMaskFast:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
                "radius_x": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1023,
                    "step": 1
                }),
                "radius_y": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 1023,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "blur_mask"

    CATEGORY = "mask/filters"

    def blur_mask(self, masks, radius_x, radius_y):
        
        if radius_x + radius_y == 0:
            return (masks,)
        
        dx = radius_x * 2 + 1
        dy = radius_y * 2 + 1
        
        dup = copy.deepcopy(masks.cpu().numpy())
        
        for index, mask in enumerate(dup):
            dup[index] = cv2.GaussianBlur(mask, (dx, dy), 0)
        
        return (torch.from_numpy(dup),)

def cv_blur_tensor(images, dx, dy):
    if min(dx, dy) > 100:
        np_img = torch.nn.functional.interpolate(images.detach().clone().movedim(-1,1), scale_factor=0.1, mode='bilinear').movedim(1,-1).cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx // 20 * 2 + 1, dy // 20 * 2 + 1), 0)
        return torch.nn.functional.interpolate(torch.from_numpy(np_img).movedim(-1,1), size=(images.shape[1], images.shape[2]), mode='bilinear').movedim(1,-1)
    else:
        np_img = images.detach().clone().cpu().numpy()
        for index, image in enumerate(np_img):
            np_img[index] = cv2.GaussianBlur(image, (dx, dy), 0)
        return torch.from_numpy(np_img)

class ColorMatchImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "reference": ("IMAGE", ),
                "blur": ("INT", {"default": 0, "min": 0, "max": 1023}),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"

    CATEGORY = "image/filters"

    def batch_normalize(self, images, reference, blur, factor):
        t = images.detach().clone()
        ref = reference.detach().clone()
        if ref.shape[0] < t.shape[0]:
            ref = ref[0].unsqueeze(0).repeat(t.shape[0], 1, 1, 1)
        
        if blur == 0:
            mean = torch.mean(t, (1,2), keepdim=True)
            mean_ref = torch.mean(ref, (1,2), keepdim=True)
            for i in range(t.shape[0]):
                for c in range(3):
                    t[i,:,:,c] /= mean[i,0,0,c]
                    t[i,:,:,c] *= mean_ref[i,0,0,c]
        else:
            d = blur * 2 + 1
            blurred = cv_blur_tensor(torch.clamp(t, 0.001, 1), d, d)
            blurred_ref = cv_blur_tensor(torch.clamp(ref, 0.001, 1), d, d)
            for i in range(t.shape[0]):
                for c in range(3):
                    t[i,:,:,c] /= blurred[i,:,:,c]
                    t[i,:,:,c] *= blurred_ref[i,:,:,c]
        
        t = torch.lerp(images, t, factor)
        return (t,)

class DilateErodeMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "masks": ("MASK",),
                "radius": ("INT", {
                    "default": 0,
                    "min": -1023,
                    "max": 1023,
                    "step": 1
                }),
                "shape": (["box", "circle"],),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "dilate_mask"

    CATEGORY = "mask/filters"

    def dilate_mask(self, masks, radius, shape):
        
        if radius == 0:
            return (masks,)
        
        s = abs(radius)
        d = s * 2 + 1
        k = np.zeros((d, d), np.uint8)
        if shape == "circle":
            k = cv2.circle(k, (s,s), s, 1, -1)
        else:
            k += 1
        
        dup = copy.deepcopy(masks.cpu().numpy())
        
        for index, mask in enumerate(dup):
            if radius > 0:
                dup[index] = cv2.dilate(mask, k, iterations=1)
            else:
                dup[index] = cv2.erode(mask, k, iterations=1)
        
        return (torch.from_numpy(dup),)

class EnhanceDetail:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filter_radius": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.01
                }),
                "denoise": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "detail_mult": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "enhance"

    CATEGORY = "image/filters"

    def enhance(self, images: torch.Tensor, filter_radius: int, sigma: float, denoise: float, detail_mult: float):
        
        if filter_radius == 0:
            return (images,)
        
        d = filter_radius * 2 + 1
        s = sigma / 10
        n = denoise / 10
        
        dup = copy.deepcopy(images.cpu().numpy())
        
        for index, image in enumerate(dup):
            imgB = image
            if denoise>0.0:
                imgB = cv2.bilateralFilter(image, d, n, d)
            
            imgG = np.clip(guidedFilter(image, image, d, s), 0.001, 1)
            
            details = (imgB/imgG - 1) * detail_mult + 1
            dup[index] = np.clip(details*imgG - imgB + image, 0, 1)
        
        return (torch.from_numpy(dup),)

class GuidedFilterAlpha:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "alpha": ("IMAGE",),
                "filter_radius": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 64,
                    "step": 1
                }),
                "sigma": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "guided_filter_alpha"

    CATEGORY = "image/filters"

    def guided_filter_alpha(self, images: torch.Tensor, alpha: torch.Tensor, filter_radius: int, sigma: float):
        
        d = filter_radius * 2 + 1
        s = sigma / 10
        
        i_dup = copy.deepcopy(images.cpu().numpy())
        a_dup = copy.deepcopy(alpha.cpu().numpy())
        
        for index, image in enumerate(i_dup):
            alpha_work = a_dup[index]
            i_dup[index] = guidedFilter(image, alpha_work, d, s)
        
        return (torch.from_numpy(i_dup),)

class RemapRange:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blackpoint": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "whitepoint": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap"

    CATEGORY = "image/filters"

    def remap(self, image: torch.Tensor, blackpoint: float, whitepoint: float):
        
        bp = min(blackpoint, whitepoint - 0.001)
        scale = 1 / (whitepoint - bp)
        
        i_dup = copy.deepcopy(image.cpu().numpy())
        i_dup = np.clip((i_dup - bp) * scale, 0.0, 1.0)
        
        return (torch.from_numpy(i_dup),)

Channel_List = ["red", "green", "blue", "alpha", "white", "black"]
Alpha_List = ["red", "green", "blue", "alpha", "white", "black", "none"]
class ShuffleChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": (Channel_List, {"default": "red"}),
                "green": (Channel_List, {"default": "green"}),
                "blue": (Channel_List, {"default": "blue"}),
                "alpha": (Alpha_List, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "shuffle"

    CATEGORY = "image/filters"

    def shuffle(self, image, red, green, blue, alpha):
        ch = 3 if alpha == "none" else 4
        t = torch.zeros((image.shape[0], image.shape[1], image.shape[2], ch), dtype=image.dtype, device=image.device)
        image_copy = image.detach().clone()
        
        ch_key = [red, green, blue, alpha]
        for i in range(ch):
            if ch_key[i] == "white":
                t[:,:,:,i] = 1
            elif ch_key[i] == "red":
                t[:,:,:,i] = image_copy[:,:,:,0]
            elif ch_key[i] == "green":
                t[:,:,:,i] = image_copy[:,:,:,1]
            elif ch_key[i] == "blue":
                t[:,:,:,i] = image_copy[:,:,:,2]
            elif ch_key[i] == "alpha":
                if image.shape[3] > 3:
                    t[:,:,:,i] = image_copy[:,:,:,3]
                else:
                    t[:,:,:,i] = 1
        
        return(t,)

class ClampOutliers:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "std_dev": ("FLOAT", {"default": 3.0, "min": 0.1, "max": 100.0, "step": 0.1,  "round": 0.1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "clamp_outliers"

    CATEGORY = "latent/filters"

    def clamp_outliers(self, latents, std_dev):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"]
        
        for i, latent in enumerate(t):
            for j, channel in enumerate(latent):
                sd, mean = torch.std_mean(channel, dim=None)
                t[i,j] = torch.clamp(channel, min = -sd * std_dev + mean, max = sd * std_dev + mean)
        
        latents_copy["samples"] = t
        return (latents_copy,)

class AdainLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "reference": ("LATENT", ),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "latent/filters"

    def batch_normalize(self, latents, reference, factor):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"] # [B x C x H x W]
        
        t = t.movedim(0,1) # [C x B x H x W]
        for c in range(t.size(0)):
            for i in range(t.size(1)):
                r_sd, r_mean = torch.std_mean(reference["samples"][i, c], dim=None) # index by original dim order
                i_sd, i_mean = torch.std_mean(t[c, i], dim=None)
                
                t[c, i] = ((t[c, i] - i_mean) / i_sd) * r_sd + r_mean
        
        latents_copy["samples"] = torch.lerp(latents["samples"], t.movedim(1,0), factor) # [B x C x H x W]
        return (latents_copy,)

class AdainImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "reference": ("IMAGE", ),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"

    CATEGORY = "image/filters"

    def batch_normalize(self, images, reference, factor):
        t = copy.deepcopy(images) # [B x H x W x C]
        
        t = t.movedim(-1,0) # [C x B x H x W]
        for c in range(t.size(0)):
            for i in range(t.size(1)):
                r_sd, r_mean = torch.std_mean(reference[i, :, :, c], dim=None) # index by original dim order
                i_sd, i_mean = torch.std_mean(t[c, i], dim=None)
                
                t[c, i] = ((t[c, i] - i_mean) / i_sd) * r_sd + r_mean
        
        t = torch.lerp(images, t.movedim(0,-1), factor) # [B x H x W x C]
        return (t,)

class BatchNormalizeLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "latent/filters"

    def batch_normalize(self, latents, factor):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"] # [B x C x H x W]
        
        t = t.movedim(0,1) # [C x B x H x W]
        for c in range(t.size(0)):
            c_sd, c_mean = torch.std_mean(t[c], dim=None)
            
            for i in range(t.size(1)):
                i_sd, i_mean = torch.std_mean(t[c, i], dim=None)
                
                t[c, i] = (t[c, i] - i_mean) / i_sd
            
            t[c] = t[c] * c_sd + c_mean
        
        latents_copy["samples"] = torch.lerp(latents["samples"], t.movedim(1,0), factor) # [B x C x H x W]
        return (latents_copy,)

class BatchNormalizeImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch_normalize"

    CATEGORY = "image/filters"

    def batch_normalize(self, images, factor):
        t = copy.deepcopy(images) # [B x H x W x C]
        
        t = t.movedim(-1,0) # [C x B x H x W]
        for c in range(t.size(0)):
            c_sd, c_mean = torch.std_mean(t[c], dim=None)
            
            for i in range(t.size(1)):
                i_sd, i_mean = torch.std_mean(t[c, i], dim=None)
                
                t[c, i] = (t[c, i] - i_mean) / i_sd
            
            t[c] = t[c] * c_sd + c_mean
        
        t = torch.lerp(images, t.movedim(0,-1), factor) # [B x H x W x C]
        return (t,)

class DifferenceChecker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images1": ("IMAGE", ),
                "images2": ("IMAGE", ),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "difference_checker"

    CATEGORY = "image/filters"

    def difference_checker(self, images1, images2, multiplier):
        t = copy.deepcopy(images1)
        t = torch.abs(images1 - images2) * multiplier
        return (torch.clamp(t, min=0, max=1),)

class ImageConstant:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "red": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "green": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "blue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image/filters"

    def generate(self, width, height, batch_size, red, green, blue):
        r = torch.full([batch_size, height, width, 1], red)
        g = torch.full([batch_size, height, width, 1], green)
        b = torch.full([batch_size, height, width, 1], blue)
        return (torch.cat((r, g, b), dim=-1), )

def hsv_to_rgb(h, s, v):
    if s:
        if h == 1.0: h = 0.0
        i = int(h*6.0)
        f = h*6.0 - i
        
        w = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        
        if i==0: return (v, t, w)
        if i==1: return (q, v, w)
        if i==2: return (w, v, t)
        if i==3: return (w, q, v)
        if i==4: return (t, w, v)
        if i==5: return (v, w, q)
    else: return (v, v, v)

class ImageConstantHSV:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "hue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "saturation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              "value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"

    CATEGORY = "image/filters"

    def generate(self, width, height, batch_size, hue, saturation, value):
        red, green, blue = hsv_to_rgb(hue, saturation, value)
        
        r = torch.full([batch_size, height, width, 1], red)
        g = torch.full([batch_size, height, width, 1], green)
        b = torch.full([batch_size, height, width, 1], blue)
        return (torch.cat((r, g, b), dim=-1), )

class OffsetLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                              "offset_0": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,  "round": 0.1}),
                              "offset_1": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,  "round": 0.1}),
                              "offset_2": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,  "round": 0.1}),
                              "offset_3": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.1,  "round": 0.1}),
                              }}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"

    CATEGORY = "latent"

    def generate(self, width, height, batch_size, offset_0, offset_1, offset_2, offset_3):
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        latent[:,0,:,:] = offset_0
        latent[:,1,:,:] = offset_1
        latent[:,2,:,:] = offset_2
        latent[:,3,:,:] = offset_3
        return ({"samples":latent}, )

class RelightSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "normals": ("IMAGE",),
                "x_dir": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01}),
                "y_dir": ("FLOAT", {"default": 0.0, "min": -1.5, "max": 1.5, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "relight"

    CATEGORY = "image/filters"

    def relight(self, image, normals, x_dir, y_dir, brightness):
        if image.shape[0] != normals.shape[0]:
            raise Exception("Batch size for image and normals must match")
        norm = normals.detach().clone() * 2 - 1
        norm = torch.nn.functional.interpolate(norm.movedim(-1,1), size=(image.shape[1], image.shape[2]), mode='bilinear').movedim(1,-1)
        light = torch.tensor([x_dir, y_dir, abs(1 - math.sqrt(x_dir ** 2 + y_dir ** 2) * 0.7)])
        light = torch.nn.functional.normalize(light, dim=0)
        
        diffuse = norm[:,:,:,0] * light[0] + norm[:,:,:,1] * light[1] + norm[:,:,:,2] * light[2]
        diffuse = torch.clip(diffuse.unsqueeze(3).repeat(1,1,1,3), 0, 1)
        
        relit = image.detach().clone()
        relit[:,:,:,:3] = torch.clip(relit[:,:,:,:3] * diffuse * brightness, 0, 1)
        return (relit,)

class LatentStats:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"latent": ("LATENT", ),}}

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("stats", "c0_mean", "c1_mean", "c2_mean", "c3_mean")
    FUNCTION = "notify"
    OUTPUT_NODE = True

    CATEGORY = "utils"

    def notify(self, latent):
        latents = latent["samples"]
        width, height = latents.size(3), latents.size(2)
        
        text = ["",]
        text[0] = f"batch size: {latents.size(0)}"
        text.append(f"width: {width} ({width * 8})")
        text.append(f"height: {height} ({height * 8})")
        
        cmean = [0,0,0,0]
        for i in range(4):
            minimum = torch.min(latents[:,i,:,:]).item()
            maximum = torch.max(latents[:,i,:,:]).item()
            std_dev, mean = torch.std_mean(latents[:,i,:,:], dim=None)
            cmean[i] = mean
            
            text.append(f"c{i} mean: {mean:.1f} std_dev: {std_dev:.1f} min: {minimum:.1f} max: {maximum:.1f}")
        
        
        printtext = "\033[36mLatent Stats:\033[m"
        for t in text:
            printtext += "\n    " + t
        
        returntext = ""
        for i in range(len(text)):
            if i > 0:
                returntext += "\n"
            returntext += text[i]
        
        print(printtext)
        return (returntext, cmean[0], cmean[1], cmean[2], cmean[3])

def sRGBtoLinear(npArray):
    less = npArray <= 0.0404482362771082
    npArray[less] = npArray[less] / 12.92
    npArray[~less] = np.power((npArray[~less] + 0.055) / 1.055, 2.4)

def linearToSRGB(npArray):
    less = npArray <= 0.0031308
    npArray[less] = npArray[less] * 12.92
    npArray[~less] = np.power(npArray[~less], 1/2.4) * 1.055 - 0.055

def linearToTonemap(npArray, tonemap_scale):
    npArray /= tonemap_scale
    more = npArray > 0.06
    SLog3 = np.clip((np.log10((npArray + 0.01)/0.19) * 261.5 + 420) / 1023, 0, 1)
    npArray[more] = np.power(1 / (1 + (1 / np.power(SLog3[more] / (1 - SLog3[more]), 1.7))), 1.7)
    npArray *= tonemap_scale

def tonemapToLinear(npArray, tonemap_scale):
    npArray /= tonemap_scale
    more = npArray > 0.06
    x = np.power(np.clip(npArray, 0.000001, 1), 1/1.7)
    ut = 1 / (1 + np.power((-1 / x) * (x - 1), 1/1.7))
    npArray[more] = np.power(10, (ut[more] * 1023 - 420)/261.5) * 0.19 - 0.01
    npArray *= tonemap_scale

class Tonemap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "input_mode": (["linear", "sRGB"],),
                "output_mode": (["sRGB", "linear"],),
                "tonemap_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "image/filters"

    def apply(self, images, input_mode, output_mode, tonemap_scale):
        t = images.detach().clone().cpu().numpy().astype(np.float32)
        
        if input_mode == "sRGB":
            sRGBtoLinear(t[:,:,:,:3])
        
        linearToTonemap(t[:,:,:,:3], tonemap_scale)
        
        if output_mode == "sRGB":
            linearToSRGB(t[:,:,:,:3])
            t = np.clip(t, 0, 1)
        
        t = torch.from_numpy(t)
        return (t,)

class UnTonemap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "input_mode": (["sRGB", "linear"],),
                "output_mode": (["linear", "sRGB"],),
                "tonemap_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "image/filters"

    def apply(self, images, input_mode, output_mode, tonemap_scale):
        t = images.detach().clone().cpu().numpy().astype(np.float32)
        
        if input_mode == "sRGB":
            sRGBtoLinear(t[:,:,:,:3])
        
        tonemapToLinear(t[:,:,:,:3], tonemap_scale)
        
        if output_mode == "sRGB":
            linearToSRGB(t[:,:,:,:3])
            t = np.clip(t, 0, 1)
        
        t = torch.from_numpy(t)
        return (t,)

def exposure(npArray, stops):
    more = npArray > 0
    npArray[more] *= pow(2, stops)

class ExposureAdjust:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "stops": ("FLOAT", {"default": 0.0, "min": -100, "max": 100, "step": 0.01}),
                "input_mode": (["sRGB", "linear"],),
                "output_mode": (["sRGB", "linear"],),
                "use_tonemap": ("BOOLEAN", {"default": False}),
                "tonemap_scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 10, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "image/filters"

    def apply(self, images, stops, input_mode, output_mode, use_tonemap, tonemap_scale):
        t = images.detach().clone().cpu().numpy().astype(np.float32)
        
        if input_mode == "sRGB":
            sRGBtoLinear(t[:,:,:,:3])
        
        if use_tonemap:
            tonemapToLinear(t[:,:,:,:3], tonemap_scale)
        
        exposure(t[:,:,:,:3], stops)
        
        if use_tonemap:
            linearToTonemap(t[:,:,:,:3], tonemap_scale)
        
        if output_mode == "sRGB":
            linearToSRGB(t[:,:,:,:3])
            t = np.clip(t, 0, 1)
        
        t = torch.from_numpy(t)
        return (t,)

# Normal map standard coordinates: +r:+x:right, +g:+y:up, +b:+z:in
class ConvertNormals:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "normals": ("IMAGE",),
                "input_mode": (["BAE", "MiDaS", "Standard"],),
                "output_mode": (["BAE", "MiDaS", "Standard"],),
                "scale_XY": ("FLOAT",{"default": 1, "min": 0, "max": 100, "step": 0.001}),
                "normalize": ("BOOLEAN", {"default": True}),
                "fix_black": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "optional_fill": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_normals"

    CATEGORY = "image/filters"

    def convert_normals(self, normals, input_mode, output_mode, scale_XY, normalize, fix_black, optional_fill=None):
        t = normals.detach().clone()
        
        if input_mode == "BAE":
            t[:,:,:,0] = 1 - t[:,:,:,0] # invert R
        elif input_mode == "MiDaS":
            t[:,:,:,:3] = torch.stack([1 - t[:,:,:,2], t[:,:,:,1], t[:,:,:,0]], dim=3) # BGR -> RGB and invert R
        
        if fix_black:
            key = torch.clamp(1 - t[:,:,:,2] * 2, min=0, max=1)
            if optional_fill == None:
                t[:,:,:,0] += key * 0.5
                t[:,:,:,1] += key * 0.5
                t[:,:,:,2] += key
            else:
                fill = optional_fill.detach().clone()
                if fill.shape[1:3] != t.shape[1:3]:
                    fill = torch.nn.functional.interpolate(fill.movedim(-1,1), size=(t.shape[1], t.shape[2]), mode='bilinear').movedim(1,-1)
                if fill.shape[0] != t.shape[0]:
                    fill = fill[0].unsqueeze(0).expand(t.shape[0], -1, -1, -1)
                t[:,:,:,:3] += fill[:,:,:,:3] * key.unsqueeze(3).expand(-1, -1, -1, 3)
        
        t[:,:,:,:2] = (t[:,:,:,:2] - 0.5) * scale_XY + 0.5
        
        if normalize:
            t[:,:,:,:3] = torch.nn.functional.normalize(t[:,:,:,:3] * 2 - 1, dim=3) / 2 + 0.5
        
        if output_mode == "BAE":
            t[:,:,:,0] = 1 - t[:,:,:,0] # invert R
        elif output_mode == "MiDaS":
            t[:,:,:,:3] = torch.stack([t[:,:,:,2], t[:,:,:,1], 1 - t[:,:,:,0]], dim=3) # invert R and BGR -> RGB
        
        return (t,)

class BatchAverageImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["mean", "median"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "image/filters"

    def apply(self, images, operation):
        t = images.detach().clone()
        if operation == "mean":
            return (torch.mean(t, dim=0, keepdim=True),)
        elif operation == "median":
            return (torch.median(t, dim=0, keepdim=True)[0],)
        return(t,)

class NormalMapSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "scale_XY": ("FLOAT",{"default": 1, "min": 0, "max": 100, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normal_map"

    CATEGORY = "image/filters"

    def normal_map(self, images, scale_XY):
        t = images.detach().clone().cpu().numpy().astype(np.float32)
        L = np.mean(t[:,:,:,:3], axis=3)
        for i in range(t.shape[0]):
            t[i,:,:,0] = cv2.Scharr(L[i], -1, 1, 0, cv2.BORDER_REFLECT) * -1
            t[i,:,:,1] = cv2.Scharr(L[i], -1, 0, 1, cv2.BORDER_REFLECT)
        t[:,:,:,2] = 1
        t = torch.from_numpy(t)
        t[:,:,:,:2] *= scale_XY
        t[:,:,:,:3] = torch.nn.functional.normalize(t[:,:,:,:3], dim=3) / 2 + 0.5
        return (t,)

class Keyer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["luminance", "saturation", "max", "min", "red", "green", "blue", "redscreen", "greenscreen", "bluescreen"],),
                "low": ("FLOAT",{"default": 0, "step": 0.001}),
                "high": ("FLOAT",{"default": 1, "step": 0.001}),
                "gamma": ("FLOAT",{"default": 1.0, "min": 0.001, "step": 0.001}),
                "premult": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha", "mask")
    FUNCTION = "keyer"

    CATEGORY = "image/filters"

    def keyer(self, images, operation, low, high, gamma, premult):
        t = images[:,:,:,:3].detach().clone()
        
        if operation == "luminance":
            alpha = 0.2126 * t[:,:,:,0] + 0.7152 * t[:,:,:,1] + 0.0722 * t[:,:,:,2]
        elif operation == "saturation":
            minV = torch.min(t, 3)[0]
            maxV = torch.max(t, 3)[0]
            mask = maxV != 0
            alpha = maxV
            alpha[mask] = (maxV[mask] - minV[mask]) / maxV[mask]
        elif operation == "max":
            alpha = torch.max(t, 3)[0]
        elif operation == "min":
            alpha = torch.min(t, 3)[0]
        elif operation == "red":
            alpha = t[:,:,:,0]
        elif operation == "green":
            alpha = t[:,:,:,1]
        elif operation == "blue":
            alpha = t[:,:,:,2]
        elif operation == "redscreen":
            alpha = 0.7 * (t[:,:,:,1] + t[:,:,:,2]) - t[:,:,:,0] + 1
        elif operation == "greenscreen":
            alpha = 0.7 * (t[:,:,:,0] + t[:,:,:,2]) - t[:,:,:,1] + 1
        elif operation == "bluescreen":
            alpha = 0.7 * (t[:,:,:,0] + t[:,:,:,1]) - t[:,:,:,2] + 1
        else: # should never be reached
            alpha = t[:,:,:,0] * 0
        
        if low == high:
            alpha = (alpha > high).to(t.dtype)
        else:
            alpha = (alpha - low) / (high - low)
        
        if gamma != 1.0:
            alpha = torch.pow(alpha, 1/gamma)
        alpha = torch.clamp(alpha, min=0, max=1).unsqueeze(3).repeat(1,1,1,3)
        if premult:
            t *= alpha
        return (t, alpha, alpha[:,:,:,0])

jitter_matrix = torch.Tensor([[[1, 0, 0], [0, 1, 0]], [[1, 0, 1], [0, 1, 0]], [[1, 0, 1], [0, 1, 1]],
                              [[1, 0, 0], [0, 1, 1]], [[1, 0,-1], [0, 1, 1]], [[1, 0,-1], [0, 1, 0]],
                              [[1, 0,-1], [0, 1,-1]], [[1, 0, 0], [0, 1,-1]], [[1, 0, 1], [0, 1,-1]]])

class JitterImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "jitter_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "jitter"

    CATEGORY = "image/filters/jitter"

    def jitter(self, images, jitter_scale):
        t = images.detach().clone().movedim(-1,1) # [B x C x H x W]
        
        theta = jitter_matrix.detach().clone().to(t.device)
        theta[:,0,2] *= jitter_scale * 2 / t.shape[3]
        theta[:,1,2] *= jitter_scale * 2 / t.shape[2]
        affine = torch.nn.functional.affine_grid(theta, torch.Size([9, t.shape[1], t.shape[2], t.shape[3]]))
        
        batch = []
        for i in range(t.shape[0]):
            jb = t[i].repeat(9,1,1,1)
            jb = torch.nn.functional.grid_sample(jb, affine, mode='bilinear', padding_mode='border', align_corners=None)
            batch.append(jb)
        
        t = torch.cat(batch, dim=0).movedim(1,-1) # [B x H x W x C]
        return (t,)

class UnJitterImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "jitter_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
                "oflow_align": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "jitter"

    CATEGORY = "image/filters/jitter"

    def jitter(self, images, jitter_scale, oflow_align):
        t = images.detach().clone().movedim(-1,1) # [B x C x H x W]
        
        if oflow_align:
            pbar = ProgressBar(t.shape[0] // 9)
            raft_model, raft_device = load_raft()
            batch = []
            for i in trange(t.shape[0] // 9):
                batch1 = t[i*9].unsqueeze(0).repeat(9,1,1,1)
                batch2 = t[i*9:i*9+9]
                flows = raft_flow(raft_model, raft_device, batch1, batch2)
                batch.append(flows)
                pbar.update(1)
            flows = torch.cat(batch, dim=0)
            t = flow_warp(t, flows)
        else:
            theta = jitter_matrix.detach().clone().to(t.device)
            theta[:,0,2] *= jitter_scale * -2 / t.shape[3]
            theta[:,1,2] *= jitter_scale * -2 / t.shape[2]
            affine = torch.nn.functional.affine_grid(theta, torch.Size([9, t.shape[1], t.shape[2], t.shape[3]]))
            batch = []
            for i in range(t.shape[0] // 9):
                jb = t[i*9:i*9+9]
                jb = torch.nn.functional.grid_sample(jb, affine, mode='bicubic', padding_mode='border', align_corners=None)
                batch.append(jb)
            t = torch.cat(batch, dim=0)
        
        t = t.movedim(1,-1) # [B x H x W x C]
        return (t,)

class BatchAverageUnJittered:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "operation": (["mean", "median"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    CATEGORY = "image/filters/jitter"

    def apply(self, images, operation):
        t = images.detach().clone()
        
        batch = []
        for i in range(t.shape[0] // 9):
            if operation == "mean":
                batch.append(torch.mean(t[i*9:i*9+9], dim=0, keepdim=True))
            elif operation == "median":
                batch.append(torch.median(t[i*9:i*9+9], dim=0, keepdim=True)[0])
        
        return (torch.cat(batch, dim=0),)

class BatchAlign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "ref_frame": ("INT", {"default": 0, "min": 0}),
                "direction": (["forward", "backward"],),
                "blur": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("aligned", "flow")
    FUNCTION = "apply"

    CATEGORY = "image/filters"

    def apply(self, images, ref_frame, direction, blur):
        t = images.detach().clone().movedim(-1,1) # [B x C x H x W]
        rf = min(ref_frame, t.shape[0] - 1)
        
        raft_model, raft_device = load_raft()
        ref = t[rf].unsqueeze(0).repeat(t.shape[0],1,1,1)
        if direction == "forward":
            flows = raft_flow(raft_model, raft_device, ref, t)
        else:
            flows = raft_flow(raft_model, raft_device, t, ref) * -1
        
        if blur > 0:
            d = blur * 2 + 1
            dup = flows.movedim(1,-1).detach().clone().cpu().numpy()
            blurred = []
            for img in dup:
                blurred.append(torch.from_numpy(cv2.GaussianBlur(img, (d,d), 0)).unsqueeze(0).movedim(-1,1))
            flows = torch.cat(blurred).to(flows.device)
        
        t = flow_warp(t, flows)
        
        t = t.movedim(1,-1) # [B x H x W x C]
        f = images.detach().clone() * 0
        f[:,:,:,:2] = flows.movedim(1,-1)
        return (t,f)

class InstructPixToPixConditioningAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "new": ("LATENT", ),
                             "original": ("LATENT", ),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("cond1", "cond2", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/instructpix2pix"

    def encode(self, positive, negative, new, original):
        new_shape, orig_shape = new["samples"].shape, original["samples"].shape
        if new_shape != orig_shape:
            raise Exception(f"Latent shape mismatch: {new_shape} and {orig_shape}")
        
        out_latent = {}
        out_latent["samples"] = new["samples"]
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = original["samples"]
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], negative, out_latent)

class LatentNormalizeShuffle:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "flatten": ("INT", {"default": 0, "min": 0, "max": 16}),
                "normalize": ("BOOLEAN", {"default": True}),
                "shuffle": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "latent/filters"

    def batch_normalize(self, latents, flatten, normalize, shuffle):
        latents_copy = copy.deepcopy(latents)
        t = latents_copy["samples"] # [B x C x H x W]
        
        if flatten > 0:
            d = flatten * 2 + 1
            channels = t.shape[1]
            kernel = gaussian_kernel(d, 1, device=t.device).repeat(channels, 1, 1).unsqueeze(1)
            t_blurred = torch.nn.functional.conv2d(t, kernel, padding='same', groups=channels)
            t = t - t_blurred
        
        if normalize:
            for b in range(t.shape[0]):
                for c in range(4):
                    t_sd, t_mean = torch.std_mean(t[b,c])
                    t[b,c] = (t[b,c] - t_mean) / t_sd
        
        if shuffle:
            t_shuffle = []
            for i in (1,2,3,0):
                t_shuffle.append(t[:,i])
            t = torch.stack(t_shuffle, dim=1)
        
        latents_copy["samples"] = t
        return (latents_copy,)

NODE_CLASS_MAPPINGS = {
    "AdainImage": AdainImage,
    "AdainLatent": AdainLatent,
    "AlphaClean": AlphaClean,
    "AlphaMatte": AlphaMatte,
    "BatchAlign": BatchAlign,
    "BatchAverageImage": BatchAverageImage,
    "BatchAverageUnJittered": BatchAverageUnJittered,
    "BatchNormalizeImage": BatchNormalizeImage,
    "BatchNormalizeLatent": BatchNormalizeLatent,
    "BetterFilmGrain": BetterFilmGrain,
    "BlurImageFast": BlurImageFast,
    "BlurMaskFast": BlurMaskFast,
    "ClampOutliers": ClampOutliers,
    "ColorMatchImage": ColorMatchImage,
    "ConvertNormals": ConvertNormals,
    "DifferenceChecker": DifferenceChecker,
    "DilateErodeMask": DilateErodeMask,
    "EnhanceDetail": EnhanceDetail,
    "ExposureAdjust": ExposureAdjust,
    "GuidedFilterAlpha": GuidedFilterAlpha,
    "ImageConstant": ImageConstant,
    "ImageConstantHSV": ImageConstantHSV,
    "JitterImage": JitterImage,
    "Keyer": Keyer,
    "LatentStats": LatentStats,
    "NormalMapSimple": NormalMapSimple,
    "OffsetLatentImage": OffsetLatentImage,
    "RelightSimple": RelightSimple,
    "RemapRange": RemapRange,
    "ShuffleChannels": ShuffleChannels,
    "Tonemap": Tonemap,
    "UnJitterImage": UnJitterImage,
    "UnTonemap": UnTonemap,
    "InstructPixToPixConditioningAdvanced": InstructPixToPixConditioningAdvanced,
    "LatentNormalizeShuffle": LatentNormalizeShuffle,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdainImage": "AdaIN (Image)",
    "AdainLatent": "AdaIN (Latent)",
    "AlphaClean": "Alpha Clean",
    "AlphaMatte": "Alpha Matte",
    "BatchAlign": "Batch Align (RAFT)",
    "BatchAverageImage": "Batch Average Image",
    "BatchAverageUnJittered": "Batch Average Un-Jittered",
    "BatchNormalizeImage": "Batch Normalize (Image)",
    "BatchNormalizeLatent": "Batch Normalize (Latent)",
    "BetterFilmGrain": "Better Film Grain",
    "BlurImageFast": "Blur Image (Fast)",
    "BlurMaskFast": "Blur Mask (Fast)",
    "ClampOutliers": "Clamp Outliers",
    "ColorMatchImage": "Color Match Image",
    "ConvertNormals": "Convert Normals",
    "DifferenceChecker": "Difference Checker",
    "DilateErodeMask": "Dilate/Erode Mask",
    "EnhanceDetail": "Enhance Detail",
    "ExposureAdjust": "Exposure Adjust",
    "GuidedFilterAlpha": "Guided Filter Alpha",
    "ImageConstant": "Image Constant Color (RGB)",
    "ImageConstantHSV": "Image Constant Color (HSV)",
    "JitterImage": "Jitter Image",
    "Keyer": "Keyer",
    "LatentStats": "Latent Stats",
    "NormalMapSimple": "Normal Map (Simple)",
    "OffsetLatentImage": "Offset Latent Image",
    "RelightSimple": "Relight (Simple)",
    "RemapRange": "Remap Range",
    "ShuffleChannels": "Shuffle Channels",
    "Tonemap": "Tonemap",
    "UnJitterImage": "Un-Jitter Image",
    "UnTonemap": "UnTonemap",
    "InstructPixToPixConditioningAdvanced": "InstructPixToPixConditioningAdvanced",
    "LatentNormalizeShuffle": "LatentNormalizeShuffle",
}