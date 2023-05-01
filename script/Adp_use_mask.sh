
#!/usr/bin/env bash         \

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=./experiments/Adapter/ViT-B_prompt_adapter_8_patch.yaml
CKPT=/home/shang/ViT-B_16.npz
WEIGHT_DECAY=0.0001

mkdir -p log/Adp_noise
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# cifar100 caltech101 dtd oxford_flowers102 svhn sun397 oxford_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
#  --switch_bn True --use_pre_soft --mild_l_inf 0.001 --patch_fool

for LR in 0.001
do 
    for DATASET in svhn
    do
        CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=1676 --use_env train_noise.py\
        --data-path=./data/vtab-1k/${DATASET} --data-set=${DATASET} --cfg=${CONFIG} --resume=${CKPT} --output_dir=./saves/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_adapter --batch-size=64 --lr=${LR}\
        --pretrained_noise --patch_fool --fixed_noise --epochs=100 --is_adapter --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 --direct_resize --smoothing=0 --launcher="pytorch"\
        2>&1 | tee -a log/Adp_noise/${currenttime}-${DATASET}-${LR}-Adp.log
    done
done