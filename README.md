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

### Image Constant (RGB/HSV)

Create an empty image of any color, either RGB or HSV

### Offset Latent Image

Create an empty latent image with custom values, for offset noise but with per-channel control. Can be combined with Latent Stats to get channel values.

### Latent Stats

Prints some stats about the latents (dimensions, and per-channel mean, std dev, min, and max)

### Tonemap / UnTonemap

Apply or remove a log + contrast curve tonemap

Apply tonemap:
```
power = 1.7
SLog3R = clamp((log10((r + 0.01)/0.19) * 261.5 + 420) / 1023, 0, 1)
SLog3G = clamp((log10((g + 0.01)/0.19) * 261.5 + 420) / 1023, 0, 1)
SLog3B = clamp((log10((b + 0.01)/0.19) * 261.5 + 420) / 1023, 0, 1)

r = r > 0.06 ? pow(1 / (1 + (1 / pow(SLog3R / (1 - SLog3R), power))), power) : r
g = g > 0.06 ? pow(1 / (1 + (1 / pow(SLog3G / (1 - SLog3G), power))), power) : g
b = b > 0.06 ? pow(1 / (1 + (1 / pow(SLog3B / (1 - SLog3B), power))), power) : b
```

Remove tonemap:
```
power = 1.7
SR = 1 / (1 + pow((-1/pow(r, 1/power)) * (pow(r, 1/power) - 1), 1/power))
SG = 1 / (1 + pow((-1/pow(g, 1/power)) * (pow(g, 1/power) - 1), 1/power))
SB = 1 / (1 + pow((-1/pow(b, 1/power)) * (pow(b, 1/power) - 1), 1/power))

r = r > 0.06 ? pow(10, (SR * 1023 - 420)/261.5) * 0.19 - 0.01 : r
g = g > 0.06 ? pow(10, (SG * 1023 - 420)/261.5) * 0.19 - 0.01 : g
b = b > 0.06 ? pow(10, (SB * 1023 - 420)/261.5) * 0.19 - 0.01 : b
```

### Exposure Adjust

Linear exposure adjustment in f-stops, with optional tonemap.

### Convert Normals

Translate between different normal map color spaces, with optional normalization fix and black region fix.

### Batch Average Image

Returns the single average image of a batch.

### Normal Map (Simple)

Simple high-frequency normal map from Scharr operator

### Keyer

Image keyer with luma/sat/channel/greenscreen/etc options

### JitterImage, UnJitterImage, BatchAverageUnJittered

For supersampling/antialiasing workflows.

### Shuffle

Move channels around at will.

### ColorMatch

Match image color to reference image, using mean or blur. Similar to AdaIN.

### RestoreDetail

Transfers details from one image to another using frequency separation techniques. Useful for restoring the lost details from IC-Light or other img2img workflows. Has options for add/subtract method (fewer artifacts, but mostly ignores highlights) or divide/multiply (more natural but can create artifacts in areas that go from dark to bright), and either gaussian blur or guided filter (prevents oversharpened edges).

![restore_detail](https://github.com/spacepxl/ComfyUI-Image-Filters/assets/143970342/aa4fedce-e622-4ebe-b8e7-6348d37878a5)

### BetterFilmGrain

Yet another film grain node, but this one looks better (realistic grain structure, no pixel-perfect RGB glitter, natural luminance/intensity response) and is 10x faster than the next best option (ProPostFilmGrain).
