cd src
# train, the model is finetuned from a CenterNet detection model from the CenterNet model zoo.
# python main.py tracking --exp_id coco_tracking --tracking --load_model ../models/ctdet_coco_dla_2x.pth  --gpus 0,1,2,3,4,5,6,7 --batch_size 128 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
# python main.py tracking --exp_id coco_tracking --tracking --load_model ../models/ctdet_coco_dla_2x.pth  --gpus 0 --batch_size 128 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1

# python main.py tracking --exp_id ball_tracking --tracking --gpus 0 --batch_size 1 --lr 5e-4 --num_workers 1 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
python main.py tracking --exp_id ball_tracking --tracking --load_model ../models/ctdet_coco_dla_2x.pth --gpus 0 --batch_size 2 --lr 5e-4 --num_workers 1 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1

# python main.py tracking \ 
# --exp_id coco_tracking \
# --tracking \
# --load_model ../models/coco_tracking.pth \
# --gpus 0 \
# --batch_size 1 \
# --lr 5e-4 \
# --num_workers 1 \
# --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1


# https://discourse.psychopy.org/t/subprocess-calledprocesserror-command-git-tag-returned-non-zero-exit-status-128/7926

