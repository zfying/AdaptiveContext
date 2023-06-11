set -e
cd ../
# set basic arguments - ColorObject
SAVE_DIR=./saved_models_color
DATA_DIR=../data/colorobject
DATASET=ColorObject
BIAS=10and7
# set 
GPU_NUM=0

# baseline: "ERM", "IRM", "VREx", "GroupDRO", "MLDG", "Fish"
ALGO=IRM
for SEED in 0 7 77 8 88; do
    EXP_NAME=${BIAS}_${ALGO}_seed${SEED}

    CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u train.py \
    --data_dir ${DATA_DIR} \
    --dataset ${DATASET} \
    --output_dir ${SAVE_DIR} \
    --exp_name ${EXP_NAME} \
    --algorithm ${ALGO} \
    --bias ${BIAS} \
    --seed ${SEED}
done 

# augmentation - fgRandom+bgOnly - mix_rate ablation
ALGO=ERM_augmentation
for SEED in 0 7 77 8 88 1 777 7777 888 8888; do
    for ((FG_W=0,BG_W=0.5;FG_W<=0.5;FG_W+=0.05, BG_W-=0.05))
    do
       EXP_NAME=${BIAS}_${ALGO}_fgRandom${FG_W}_bgOnly${BG_W}_seed${SEED}

        CUDA_VISIBLE_DEVICES=${GPU_NUM} python -u train.py \
        --data_dir ${DATA_DIR} \
        --dataset ${DATASET} \
        --output_dir ${SAVE_DIR} \
        --exp_name ${EXP_NAME} \
        --algorithm ${ALGO} \
        --bias ${BIAS} \
        --seed ${SEED} \
        --use_two_labels \
        --use_mask \
        --aug_fg \
        --aug_fg_type random_bg \
        --aug_bg \
        --fg_weight ${FG_W} \
        --bg_weight ${BG_W}
    done
done 