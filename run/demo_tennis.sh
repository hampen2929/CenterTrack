cd src

# demo
python demo.py \
tracking \
--load_model ../exp/tracking/tennis3_test/model_best.pth \
--demo ../videos/match_01_part.mp4 \
--track_thresh 0.4 \
--debug 0

# coco_clip1
# python demo.py \
# tracking \
# --load_model ../models/model_3.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.45 \
# --debug 1

# trained
# python demo.py \
# tracking \
# --load_model ../models/tennis3_4.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1

# trained
# python demo.py \
# tracking \
# --load_model ../models/tennis3_11.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.4

# Default
# python demo.py \
# tracking \
# --load_model ../models/coco_tracking.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1


# # --load_model ../models/coco_pose_tracking.pth \

export PYTHONPATH="/workspace/pyvino/pyvino/model/human_pose_estimation/human_3d_pose_estimator/pose_extractor/build/:$PYTHONPATH"


echo 'c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"' >> ~/.jupyter/jupyter_notebook_config.py
