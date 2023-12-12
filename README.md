# ComfyUI-Image-Filters

Image and matte filtering nodes for ComfyUI

`image/filters/*`

Two install batch files are provided, `install.bat` which only installs requirements, and `import_error_install.bat`, which uninstalls all versions of opencv-python before reinstalling only the correct version, opencv-contrib-python (use this if you get import errors relating to opencv or cv2, which are caused by having multiple versions or the wrong version of opencv installed.)

## Alpha Clean

Clean up holes and near-solid areas in a matte.

## Enhance Detail

Increase or decrease details in an image or batch of images using a guided filter (as opposed to the typical gaussian blur used by most sharpening filters.)

![enhance](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/enhance_detail.png)

## Guided Filter Alpha

Use a guided filter to feather edges of a matte based on RGB colors. Works best with a strong color separation between FG and BG.

![guidedfilteralpha](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/guided_filter_alpha.png)

## Remap Range

Fits the color range of an image to a new blackpoint and whitepoint (clamped). Useful for clamping or thresholding soft mattes.

## TODO:
- bilateral filter image for single frame denoise
- temporal bilateral filter for video denoise
- deconvolution
