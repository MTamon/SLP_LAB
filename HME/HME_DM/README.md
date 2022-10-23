# HME_DM
Data Maker for TUT reserch, Head-Motion-Estimation.

## Prepare
```Bash:
cd HME/HME_DM/modules
git clone -b dev https://github.com/MTamon/FaceMotionDetection.git
git clone https://github.com/MTamon/DataFileCollector.git
```

## Running
```Bash:
cd HME/HME_DM
python process.py --target /mnt/database/CEJC/Version1.0/data --output ../data --log ./log/hme.log --min-detection-confidence 0.7 --min-tracking-confidence 0.5 --max-num-face 1 --model-selection 1 --frame-step 1 --box-ratio 1.1 --track-volatility 0.3 --lost-volatility 0.1 --size-volatility 0.03 --sub-track-volatility 1.0 --sub-size-volatility 0.5 --threshold 0.3 --overlap 0.9 --integrate-step 1 --integrate-volatility 0.4 --use-tracking --prohibit-integrate 0.7 --size-limit-rate 4 --gc 0.03 --gc-term 100 --gc-success 0.1 --lost-track 2 --process-num 3 --result-length 1000000
```