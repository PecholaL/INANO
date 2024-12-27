@echo off
setlocal

set "PYTHON_PATH=C:\Users\leeklll\miniconda3\envs\pyT\python.exe"

%PYTHON_PATH% "C:\Users\leeklll\Documents\DL\INANO\train.py" ^
    -train_config_path train.yaml ^
    -dataset_path c:\\Users\\leeklll\\Documents\\DL\\inano_data\\spk_emb.npy ^
    -device cuda ^
    -store_model_path c:\Users\leeklll\Documents\DL\inano_data\save\ ^
    -load_model_path c:\Users\leeklll\Documents\DL\inano_data\save\ ^
    -summary_steps 2000 ^
    -save_steps 10000 ^
    -iterations 50000

endlocal
pause