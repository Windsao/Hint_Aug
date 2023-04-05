
#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=/home/sw99/NOAH/experiments/Adapter/ViT-B_prompt_adapter_8_patch.yaml
CKPT=/home/sw99/ViT-B_16.npz
WEIGHT_DECAY=0.0001


mkdir -p LoRA_TL
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for LR in 0.0005 0.0002
do 
    for DATASET in cifar100 caltech101 dtd oxford_flowers102 oxford_pet
    do
        CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr="192.168.122.1" --master_port=1672 --use_env train_patch.py --switch_bn True --mild_l_inf 0.1 --patch_fool --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_adapter --batch-size=16 --lr=${LR} --epochs=100 --is_LoRA --weight-decay=${WEIGHT_DECAY} --launcher="pytorch"\
            2>&1 | tee -a LoRA_TL/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-adapter.log
    done
done