
patience=10
gpu=0

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset ushcn --state def --history 24 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 1.5 --stride 1.5 --nhead 4  --nlayer 2 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset ushcn --state def --history 24 --pred_window 6 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 1.5 --stride 1.5 --nhead 4  --nlayer 2 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset ushcn --state def --history 24 --pred_window 12 \
    --patience $patience --batch_size 128 --lr 1e-3 \
    --patch_size 1.5 --stride 1.5 --nhead 4  --nlayer 2 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done