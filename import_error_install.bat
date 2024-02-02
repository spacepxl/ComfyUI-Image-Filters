@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo installing requirements...

if exist "%python_exec%" (
    echo Installing with ComfyUI Portable
	%python_exec% -s -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
    for /f "delims=" %%i in (%requirements_txt%) do (
        %python_exec% -s -m pip install "%%i"
    )
) else (
    echo Installing with system Python
	pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
    for /f "delims=" %%i in (%requirements_txt%) do (
        pip install "%%i"
    )
)

pause