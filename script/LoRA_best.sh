
#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=/data1/sw99/NOAH/experiments/Adapter/ViT-B_prompt_adapter_8_patch.yaml
CKPT=/data1/sw99/ViT-B_16.npz
WEIGHT_DECAY=0.0001


mkdir -p LoRA_pre_sw_pf_0.1_0.001
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for LR in 0.001
do 
    for DATASET in food-101 oxford_pets stanford_cars oxford_flowers fgvc_aircraft 
    do
        for SEED in 2
        do
            for SHOT in 4 8 16
            do 
                CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr="192.168.122.1" --master_port=1672 --use_env train_patch.py --switch_bn True --mild_l_inf 0.1 --patch_fool --data-path=./data/dataset/${DATASET} --data-set=${DATASET}-FS --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/few-shot_${DATASET}_shot-${SHOT}_seed-${SEED}_lr-${LR}_wd-${WEIGHT_DECAY}_adapter --batch-size=16 --lr=${LR} --epochs=100 --is_LoRA --weight-decay=${WEIGHT_DECAY} --few-shot-seed=${SEED} --few-shot-shot=${SHOT} --launcher="pytorch"\
                    2>&1 | tee -a LoRA_pre_sw_pf_0.1_0.001/${currenttime}-${DATASET}-${LR}-seed-${SEED}-shot-${SHOT}-adapter.log
          done
        done
    done
done