python ohta.py \
--datasite /mnt/t4/log_segments/data \
--log log/trainOH.log \
--train-result-path transition/trainO.csv \
--valid-result-path transition/validO.csv \
--use-model alphaS \
--model-type mediam \
--use-power \
--use-delta \
--use-person1 \
--use-person2 \
--batch-size 128 \
--valid-rate 0.3 \
--truncate-excess \
--do-shuffle \
--epoch 100 \
--model-save-path model/OHTA.pth \
--save-per-epoch \
--lr 1e-4 \
--beta1 0.9 \
--beta2 0.98 \
--eps 1e-8 \
--weight-decay 1e-2 