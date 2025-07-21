## ComfyUI-Image-Filters

Started as just some image processing nodes, but now more of a kitchen sink nodepack

Two install batch files are provided, `install.bat` which only installs requirements, and `import_error_install.bat`, which uninstalls all versions of opencv then reinstalls all 4 variants with matching version (use this if you get import errors relating to opencv or cv2, which are caused by manager or other node packs installing different variants and/or versions.)

Or if you want to manage requirements manually, the only opencv variant you actually need is `opencv-contrib-python`, it covers all opencv requirements.

## Nodes

<details><summary>Latent</summary>

### AdaIN Latent

Normalizes latents to the mean and std dev of a reference input. Useful for getting rid of color shift from high denoise strength, or matching color to a reference in general.

### AdaIN Filter Latent

Same as AdaIN Latent, but with a spatial filter instead of the full frame, works like a latent color match.

### Batch Normalize Latent

Normalizes each frame in a batch to the overall mean and std dev, good for removing overall brightness flickering.

### Clamp Outliers

Clamps latents that are more than n standard deviations away from the mean. Could help with fireflies or stray noise that disrupt the VAE decode.

### Upscale Hunyuan3Dv2 Latent By

Nearest Neighbor upscaling for Hy3D latents, might be useful for hires fix.

### Latent Normalize/Shuffle

Can help break up residual image information in inversion noise.

### RandnLikeLatent

Create random noise in the same shape as the input latent, works with any latent. Useful for noise injection or other times when you just want to control noise manually.

### Offset Latent Image

Create an empty latent image with custom values, for offset noise with per-channel control. Can be combined with Latent Stats to get channel values.

### Sharpen Filter (Latent)

Increases local contrast between latent "pixels" with an image sharpening filter.

</details>

<details><summary>Image</summary>

### AdaIN Image

Normalizes images to the mean and std dev of a reference input. Useful for getting rid of color shift from high denoise strength, or matching color to a reference in general.

### Batch Align (RAFT)

Use RAFT motion vectors to warp align images

### Batch Average Image

Returns the single average image of a batch.

### Batch Normalize Image

Normalizes each frame in a batch to the overall mean and std dev, good for removing overall brightness flickering.

### BetterFilmGrain

Yet another film grain node, but this one looks better (realistic grain structure, no pixel-perfect RGB glitter, natural luminance/intensity response) and is 10x faster than the next best option (ProPostFilmGrain).

### Bilateral Filter Image

Applies a bilateral filter, can be used to remove noise or high frequency details while preserving edges

### Blur Image (Fast)

Blurs images using opencv gaussian blur, which is >100x faster than comfy image blur. Supports larger blur radius, and separate x/y controls.

### Clamp Image

Clamps image values outside of blackpoint/whitepoint range

### Color Match Image

Match image color to reference image, using overall mean or blurred image (frequency separation)

### Convert Normals

Translate between different normal map color spaces, with optional normalization fix and black region fix.

### Depth to Normals

Converts depthmap to normal map

### Difference Checker

Absolute value of the difference between inputs, with a multiplier to boost dark values for easier viewing. Alternative to the vanilla merge difference node, which is only subtraction without the abs()

### Enhance Detail

Increase or decrease details in an image or batch of images using a guided filter (as opposed to the typical gaussian blur used by most sharpening filters)

### Exposure Adjust

Linear exposure adjustment in f-stops, with optional tonemap.

### Frequency Separate/Combine

For manual frequency separation workflows

### Game of Life

Runs the Game of Life simulation with optional mask input for starting condition

### Guided Filter Image

Use a guided filter to blur an image or mask based on RGB color similarity. Works best with a strong color separation between FG and BG.

### Image Constant Color (RGB/HSV)

Create images of any solid color, from either RGB or HSV values

### Image Matting

Takes an image and trimap/mask, and refines the matte edges with [closed-form matting](https://github.com/pymatting/pymatting). Optionally extracts the foreground and background colors as well. Good for cleaning up SAM segments or hand drawn masks.

### Keyer

Basic image keyer with luma/sat/channel/greenscreen/etc options

### Median Filter Image

Applies a median filter to remove high frequency information from images, useful for frequency separation workflows

### Normal Map (Simple)

Simple high-frequency normal map from Scharr operator

### Relight (Simple)

Basic dot product (Lambertian) relighting from a normal map

### Remap Range

Fits the color range of an image to a new blackpoint and whitepoint (clamped)

### Restore Detail

Transfers details from one image to another using frequency separation. Useful for restoring the lost details from IC-Light or other img2img workflows. Has options for add/subtract method (fewer artifacts, but mostly ignores highlights) or divide/multiply (more natural but can create artifacts in areas that go from dark to bright), and either gaussian blur or guided filter (prevents oversharpened edges)

### Shuffle Channels

Move channels around at will.

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

### JitterImage, UnJitterImage, BatchAverageUnJittered

For supersampling/antialiasing workflows.

### Extract N Frames, Merge Frames By Index

For processing a smaller number of frames evenly distributed across a larger batch/video, then merging them back into the full batch

</details>

<details><summary>Mask</summary>

### Blur Mask (Fast)

Same as Blur Image (Fast) but for masks instead of images.

### Dilate/Erode Mask

Dilate or erode masks, with either a box or circle filter.

### Mask Clean

Clean up holes and near-solid areas in a matte.

### Pack Video Mask

Compresses the frames of a video mask to match video VAE latent frames, to work around comfyui's naive temporal resizing of masks.

</details>

<details><summary>Conditioning</summary>

### Conditioning Subtract

Takes the difference of two text conditions, can have interesting effects that are different from negative prompts.

### Inpaint Condition Encode/Apply

Separates the VAE encode from the conditioning so you don't have to re-encode latents every time you change a prompt.

### IP2P Conditioning Advanced

Separates the VAE encode from the conditioning so you don't have to re-encode latents every time you change a prompt.

</details>

<details><summary>Sampling</summary>

### Custom Noise

Use any latent as the noise for SamplerCustomAdvanced.

</details>

<details><summary>Utils</summary>

### Latent Stats

Get/print some stats about the latents (dimensions, and per-channel mean, std dev, min, and max)

### Model Test

Debugging node for examining model structure

### Print Sigmas

Prints the noise schedule sigma values to see what a scheduler is doing

### Visualize Latents

Shows the latent channels as a grid image

</details>
