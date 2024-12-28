@echo off
setlocal enabledelayedexpansion

set "PYTHON_PATH=C:\Users\leeklll\miniconda3\envs\pyT\python.exe"

set "ORIG_PATH=c:\Users\leeklll\Documents\DL\datasets\archive\VCTK-Corpus\VCTK-Corpus\wav48\p225\p225_001.wav"
set "OUTPUT_PATH=c:\Users\leeklll\Documents\DL\inano_data\inference"

%PYTHON_PATH% INANO.py -s "!ORIG_PATH!" -o "!OUTPUT_PATH!"

endlocal
pause