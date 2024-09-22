
patience=10
gpu=1


for seed in {1..5}
do
    python train_forecasting.py \
    --dataset activity --state def --history 3000  \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 750 --stride 750 --nhead 1 --nlayer 3 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset activity --state def --history 2000  \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 500 --stride 500 --nhead 1  --nlayer 3 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done

for seed in {1..5}
do
    python train_forecasting.py \
    --dataset activity --state def --history 1000  \
    --patience 10 --batch_size 32 --lr 1e-3 \
    --patch_size 250 --stride 250 --nhead 1  --nlayer 3 \
    --hid_dim 64 \
    --seed $seed --gpu $gpu --alpha 1
done