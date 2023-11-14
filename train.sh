python -m torch.distributed.launch --nproc_per_node=4 --master_port 46379 train_jem_acce.py \
 --batch_size 48  --weight_classify 1. \
#  --debug 