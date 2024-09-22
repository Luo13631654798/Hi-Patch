
patience=10
gpu=0


for seed in {1..5}
do
    python train_forecasting.py \
    --dataset mimic --state def --history 24 \
    --patience 10 --batch_size 8 --lr 1e-3 \
    --patch_size 12 --stride 12 --nhead 1  --nlayer 1 \
    --hid_dim 128 \
    --seed $seed --gpu $gpu --alpha 0.9
done

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset mimic --state def --history 36 \
    --patience 10 --batch_size 8 --lr 1e-3 \
    --patch_size 4.5 --stride 4.5 --nhead 1  --nlayer 1 \
    --hid_dim 8 \
    --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset mimic --state def --history 12 \
    --patience 10 --batch_size 16 --lr 1e-3 \
    --patch_size 6 --stride 6 --nhead 1  --nlayer 1 \
    --hid_dim 128 \
    --seed $seed --gpu $gpu --alpha 1
done
