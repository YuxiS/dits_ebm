source /remote-home/yfsong/.bashrc
conda activate Lora
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --main_process_port 35764 train_latent_ema.py \
 --batch_size 36  --weight_classify 0.005 --wandb_name Cifar10 \
 --in_channels 3 --num_class 10 --test_batch_size 16 --epochs 500 --base_model DiT-S/2  \
 --data_root ./data \
#  --debug
