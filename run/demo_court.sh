cd src

# python demo.py \
# tracking \
# --load_model ../exp/tracking/court_doubles_edges/model_30.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1

python demo.py \
tracking \
--load_model ../exp/tracking/court_doubles_edges_lr/model_11.pth \
--demo ../videos/match_01_part.mp4 \
--track_thresh 0.1

# python demo.py \
# tracking \
# --load_model ../models/centernet_court.pth \
# --demo ../videos/match_01_part.mp4 \
# --track_thresh 0.1

