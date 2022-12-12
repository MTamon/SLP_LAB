python main.py \
--datasite ../data/test/segments/data \
--log log/train.log \
--transition-path transition/train.csv \
--ac-feature-size 80 \
--ac-feature-width 32 \
--lstm-dim 16 \
--lstm-input-dim 32 \
--lstm-output-dim 16 \
--num-layer 2 \
--ac-linear-dim 8 \
--batch-size 128 \
--valid-rate 0.3 \
--truncate-excess \
--do-shuffle \
--epoch 20 \
--model-save-path model/_model.pth \
--save-per-epoch \
--lr 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--eps 1e-8 \
--weight-decay 1e-6