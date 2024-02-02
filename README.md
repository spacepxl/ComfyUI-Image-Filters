# ComfyUI-Image-Filters

Image and matte filtering nodes for ComfyUI

```
latent/filters/*
image/filters/*
mask/filters/*
```

Two install batch files are provided, `install.bat` which only installs requirements, and `import_error_install.bat`, which uninstalls all versions of opencv-python before reinstalling only the correct version, opencv-contrib-python (use this if you get import errors relating to opencv or cv2, which are caused by having multiple versions or the wrong version of opencv installed.)

## Nodes

### Alpha Clean

Clean up holes and near-solid areas in a matte.

### Alpha Matte

Takes an image and alpha or trimap, and refines the edges with closed-form matting. Optionally extracts the foreground and background colors as well. Good for cleaning up SAM segments or hand drawn masks.

![alphamatte](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/workflow_images/alpha_matte.png)

### Blur Image (Fast)

Blurs images using opencv gaussian blur, which is >100x faster than comfy image blur. Supports larger blur radius, and separate x/y controls.

### Blur Mask (Fast)

Same as Blur Image (Fast) but for masks instead of images.

### Dilate/Erode Mask

Dilate or erode masks, with either a box or circle filter.

### Enhance Detail

Increase or decrease details in an image or batch of images using a guided filter (as opposed to the typical gaussian blur used by most sharpening filters.)

![enhance](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/workflow_images/enhance_detail.png)

### Guided Filter Alpha

Use a guided filter to feather edges of a matte based on similar RGB colors. Works best with a strong color separation between FG and BG.

![guidedfilteralpha](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/workflow_images/guided_filter_alpha.png)

### Remap Range

Fits the color range of an image to a new blackpoint and whitepoint (clamped). Useful for clamping or thresholding soft mattes.

### Clamp Outliers

Clamps latents that are more than n standard deviations away from 0. Could help with fireflies or stray noise that disrupt the VAE decode.

### AdaIN Latent/Image

Normalizes latents/images to the mean and std dev of a reference input. Useful for getting rid of color shift from high denoise strength, or matching color to a reference in general.

### Batch Normalize Latent/Image

Normalizes each frame in a batch to the overall mean and std dev, good for removing overall brightness flickering.

### Difference Checker

Absolute value of the difference between inputs, with a multiplier to boost dark values for easier viewing. Alternative to the vanilla merge difference node, which is only a subtraction without the abs()

### Image Constant

Creates an empty image of any color, with a color picker UI.

### Offset Latent Image

Creates an empty latent image with custom values, for offset noise but with per-channel control.

### Latent Stats

Prints some stats about the latents (dimensions, and per-channel mean, std dev, min, and max)

## TODO:
- bilateral filter image for single frame denoise
- temporal filters for video denoise
- deconvolution
