#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=/home/sw99/NOAH/experiments/Adapter/ViT-B_prompt_adapter_8_patch.yaml ###########       CARE SETTING      ############
CKPT=/home/sw99/ViT-B_16.npz
WEIGHT_DECAY=0.0001


mkdir -p debug
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft 

for LR in 0.005
do 
    for DATASET in food-101
    do
        for SEED in 2
        do
            for SHOT in 4
            do 
                CUDA_VISIBLE_DEVICES=0 python train_test2.py --data-path=./data/dataset/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_adapter --batch-size=16 --lr=${LR} --epochs=100 --is_adapter --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="none"\
                    2>&1 | tee -a debug/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-adapter.log
          done
        done
    done
done
#-m torch.distributed.launch --nproc_per_node=4 --master_addr="192.168.122.1" --master_port=1665 --use_env 