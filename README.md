# ComfyUI-Image-Filters

Image filtering nodes for ComfyUI

Two install batch files are provided, install.bat which only installs requirements, and import_error_install.bat, which uninstalls all versions of opencv-python before reinstalling only the correct version, opencv-contrib-python (use this if you get import errors relating to opencv or cv2, which are caused by having multiple versions or the wrong version of opencv installed)

## Enhance Detail

Increase or decrease details in an image or batch of images using a guided filter (as opposed to the typical gaussian blur used by most sharpening filters)

![enhance](https://github.com/spacepxl/ComfyUI-Image-Filters/blob/main/enhance_detail.png)

## TODO:
- guided filter for seg/mask
- bilateral filter image for single frame denoise
- temporal bilateral filter for video denoise
- deconvolution
