python mfcc_building.py \
--target ~/mikawa/data \
--output /datasets/mikawa/new_segments \
--log log/segment.log \
--segment-size 15 \
--segment-stride 5 \
--segment-min-size 10 \
--use-feature fbank \
--video-fps 29.97 \
--sample-frequency 16000 \
--frame-length 32 \
--frame-shift 13 \
--num-mel-bins 80 \
--num-ceps 13 \
--low-frequency 20 \
--high-frequency 8000 \
--dither 1.0 \
--proc-num 7 \
--convert-path -3