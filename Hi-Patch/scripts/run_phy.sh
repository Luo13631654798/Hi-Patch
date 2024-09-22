
patience=10
gpu=0

for seed in {1..5}
do
    python train_forecasting.py --dataset physionet --state def --history 24 \
    --patience 10 --batch_size 64 --lr 1e-3 \
    --patch_size 6 --stride 6 --nhead 1  --nlayer 1 \
    --hid_dim 64 --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py --dataset physionet --state def --history 36 \
    --patience 10 --batch_size 64 --lr 1e-3 \
    --patch_size 9 --stride 9 --nhead 1  --nlayer 1 \
    --hid_dim 64 --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py --dataset physionet --state def --history 12 \
    --patience 10 --batch_size 64 --lr 1e-3 \
    --patch_size 6 --stride 6 --nhead 1  --nlayer 1 \
    --hid_dim 64 --seed $seed --gpu $gpu --alpha 1
done
