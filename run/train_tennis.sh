cd src

python main.py \
tracking \
--exp_id tennis_dataset_test \
--load_model ../models/coco_tracking.pth \
--dataset tennis \
--gpus 0 \
--batch_size 2 \
--lr 5e-5 \
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
--data_name tennis_all

# --num_classes 1
