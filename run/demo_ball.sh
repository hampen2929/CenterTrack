cd src

# demo
python demo.py \
tracking \
--load_model ../exp/tracking/ball_lr-5e-5/model_4.pth \
--demo ../videos/match_01_part.mp4 \
--track_thresh 0.1

# coco_clip1
# python demo.py \
# tracking \
# --load_model ../exp/tracking/court_test_no_num_classes/model_7.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1 \
# --debug 1

# trained
# python demo.py \
# tracking \
# --load_model ../models/ball_tracking.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1

# Default
# python demo.py \
# tracking \
# --load_model ../models/coco_tracking.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1


# # --load_model ../models/coco_pose_tracking.pth \
