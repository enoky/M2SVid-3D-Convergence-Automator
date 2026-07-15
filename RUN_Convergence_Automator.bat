@echo off
:: Set CUDA_PATH to your CUDA 13.x installation directory.
:: Note: the PyTorch cu130 wheels bundle their own CUDA runtime, so these two
:: lines are only needed if you rely on a system CUDA toolkit on PATH.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"

:: Prepend CUDA bin and libnvvp folders to PATH so they take precedence
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

call venv\scripts\activate.bat
call python app.py
pause
