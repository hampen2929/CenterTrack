cd src

# demo
# python demo.py \
# tracking \
# --load_model ../exp/tracking/ball_lr-5e-5/model_4.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1 \
# --debug 3

# coco_clip1
# python demo.py \
# tracking \
# --load_model ../models/model_3.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.45 \
# --debug 1

# trained
python demo.py \
tracking \
--load_model ../models/tennis_test_3.pth \
--demo ../videos/match_01_part.mp4 \
--track_thresh 0.1

# Default
# python demo.py \
# tracking \
# --load_model ../models/coco_tracking.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1


# # --load_model ../models/coco_pose_tracking.pth \

export PYTHONPATH="/workspace/pyvino/pyvino/model/human_pose_estimation/human_3d_pose_estimator/pose_extractor/build/:$PYTHONPATH"
