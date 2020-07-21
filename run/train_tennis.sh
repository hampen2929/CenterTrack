exp_id='ex'

cd src

# python main.py \
# tracking \
# --exp_id $exp_id \
# --load_model ../models/coco_tracking.pth \
# --gpus 0 \
# --batch_size 8 \
# --lr 5e-5 \
# --early_stopping 5 \
# --num_epochs 120 \
# --num_workers 1 \
# --pre_hm \
# --shift 0.05 \
# --scale 0.05 \
# --hm_disturb 0.05 \
# --lost_disturb 0.4 \
# --fp_disturb 0.1 \
# --val_intervals 1 \
# --save_all \
# --data_name tennis3

python main.py \
tracking \
--exp_id $exp_id \
--gpus 0 \
--batch_size 8 \
--lr 5e-5 \
--early_stopping 5 \
--num_epochs 120 \
--num_workers 1 \
--pre_hm \
--shift 0.05 \
--scale 0.05 \
--hm_disturb 0.05 \
--lost_disturb 0.4 \
--fp_disturb 0.1 \
--val_intervals 1 \
--save_all \
--data_name tennis3
# --arch dlav0_34 \

# --print_iter 1 \

python demo.py \
tracking \
--load_model ../exp/tracking/$exp_id/model_best.pth \
--demo ../videos/tennis_sample.mp4 \
--track_thresh 0.1 \
--save_video \
--debug 4 \
--video_w 1280 \
--video_h 720
