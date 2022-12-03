python process.py \
--target /mnt/database/CEJC/Version1.0/data \
--output ../data \
--log ./log/hme.log \
--min-detection-confidence 0.7 \
--min-tracking-confidence 0.5 \
--max-num-face 1 \
--model-selection 1 \
--frame-step 1 \
--box-ratio 1.1 \
--track-volatility 0.3 \
--lost-volatility 0.1 \
--size-volatility 0.03 \
--sub-track-volatility 1.0 \
--sub-size-volatility 0.5 \
--threshold 0.1 \
--overlap 0.8 \
--integrate-step 1 \
--integrate-volatility 0.4 \
--use-tracking \
--prohibit-integrate 0.7 \
--size-limit-rate 4 \
--gc 0.03 \
--gc-term 100 \
--gc-success 0.1 \
--lost-track 2 \
--process-num 3 \
--result-length 1000000 

send-slack-msg "Finish First Analysis Phase."

python post_process.py \
--original /mnt/database/CEJC/Version1.0/data \
--hme-result ../data/data \
--log log/match.log \
--order 7 \
--noise-subtract 0.2 \
--mask-subtract 0.05 \
--batch-size 5 \
--threshold-len 60 \
--threshold-use 0.2 \
--measure-method vertical 

send-slack-msg "Finish Second Analysis Phase."
send-slack-msg "Finish All Analysis Phase."