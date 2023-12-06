import torch
import os
import sys

import numpy as np
import cv2
from cv2.ximgproc import guidedFilter
import copy


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
                    "default": 0.01,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01
                }),
                "denoise": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
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
        
        dup = copy.deepcopy(images.cpu().numpy())
        
        for i in range(len(dup)):
            image = dup[i]
            imgB = image
            if denoise>0.0:
                imgB = cv2.bilateralFilter(image, d, denoise, d)
            
            imgG = guidedFilter(image, image, d, sigma)
            
            details = (imgB/imgG - 1) * detail_mult + 1
            dup[i] = details*imgG - imgB + image
        
        return (torch.from_numpy(dup),)


NODE_CLASS_MAPPINGS = {
    #"SaveTiff": SaveTiff,
    "EnhanceDetail": EnhanceDetail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #"SaveTiff": "Save Tiff",
    "EnhanceDetail": "Enhance Detail",
}