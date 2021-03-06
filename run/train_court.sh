cd src

python main.py \
tracking \
--exp_id court_doubles_edges_lr \
--load_model ../models/coco_tracking.pth \
--gpus 0 \
--batch_size 2 \
--lr 5e-6 \
--num_epochs 30 \
--num_workers 1 \
--pre_hm \
--shift 0.05 \
--scale 0.05 \
--hm_disturb 0.05 \
--lost_disturb 0.4 \
--fp_disturb 0.1 \
--val_intervals 1 \
--save_all \
--data_name court

#  \
# --num_classes 4
