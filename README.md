# ComfyUI-Image-Filters

Image and matte filtering nodes for ComfyUI

```
image/filters/*
mask/filters/*
```

Two install batch files are provided, `install.bat` which only installs requirements, and `import_error_install.bat`, which uninstalls all versions of opencv-python before reinstalling only the correct version, opencv-contrib-python (use this if you get import errors relating to opencv or cv2, which are caused by having multiple versions or the wrong version of opencv installed.)

## Alpha Clean

Clean up holes and near-solid areas in a matte.

## Blur Image (Fast)

Blurs images using opencv gaussian blur, which is >100x faster than comfy image blur. Supports larger blur radius, and separate x/y controls.

## Blur Mask (Fast)

Same as Blur Image (Fast) but for masks instead of images.

## Dilate/Erode Mask

Dilate or erode masks, with either a box or circle filter.

## Enhance Detail

Increase or decrease details in an image or batch of images using a guided filter (as opposed to the typical gaussian blur used by most sharpening filters.)

![enhance](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/enhance_detail.png)

## Guided Filter Alpha

Use a guided filter to feather edges of a matte based on similar RGB colors. Works best with a strong color separation between FG and BG.

![guidedfilteralpha](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/guided_filter_alpha.png)

## Remap Range

Fits the color range of an image to a new blackpoint and whitepoint (clamped). Useful for clamping or thresholding soft mattes.

## TODO:
- bilateral filter image for single frame denoise
- temporal bilateral filter for video denoise
- deconvolution
