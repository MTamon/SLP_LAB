python main.py \
--datasite /datasets/mikawa/segments/all \
--log log/train.log \
--train-result-path transition/train.csv \
--valid-result-path transition/valid.csv \
--ac-feature-size 80 \
--ac-feature-width 32 \
--lstm-dim 1024 \
--lstm-input-dim 2048 \
--lstm-output-dim 2048 \
--num-layer 2 \
--ac-linear-dim 512 \
--pos-feature-size 512 \
--batch-size 128 \
--valid-rate 0.3 \
--truncate-excess \
--do-shuffle \
--epoch 50 \
--model-save-path model/_model.pth \
--lr 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--eps 1e-8 \
--weight-decay 1e-2