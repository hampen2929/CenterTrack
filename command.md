# Env
docker run --runtime=nvidia -it \
--name dev_gpu \
-e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $HOME/workspace:/workspace \
-p 8888:8888 \
dev_image_gpu \
/bin/bash

docker start run dev_gpu
docker exec -it dev_gpu bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

<!-- conda create --name CenterTrack --clone idp
source activate CenterTrack -->

<!-- conda install pytorch torchvision -c pytorch -->
pip install torchvision torch --user
pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --user

CenterTrack_ROOT=/workspace/CenterTrack
# git clone --recursive https://github.com/xingyizhou/CenterTrack $CenterTrack_ROOT

pip install -r requirements.txt --user

sudo chown ubuntu:ubuntu /opt/conda/lib/python3.6/site-packages/

cd $CenterTrack_ROOT/src/lib/model/networks/
# git clone https://github.com/CharlesShang/DCNv2/ # clone if it is not automatically downloaded by `--recursive`.
cd DCNv2
./make.sh


# tracking
python ./src/demo.py tracking --load_model ./models/coco_tracking.pth --demo ./videos/match_01_part.mp4
python ./src/demo.py tracking --load_model ./models/coco_tracking.pth --demo ./videos/match_01_part.mp4 --track_thresh 0.1
python ./src/demo.py tracking --load_model ./models/ball_tracking.pth --demo ./videos/match_01_part.mp4 --track_thresh 0.1
python ./src/demo.py tracking --load_model ./models/ball_tracking2.pth --demo ./videos/match_01_part.mp4 --track_thresh 0.1


python ./src/demo.py tracking --load_model ./exp/tracking/coco_tracking/model_last.pth --demo ./videos/match_01_part.mp4 --track_thresh 0.1
python ./src/demo.py tracking, multi_pose --load_model ./models/coco_pose_tracking.pth --demo ./videos/match_01_part.mp4 --track_thresh 0.1


python ./src/demo.py tracking --load_model ./models/mot17_fulltrain.pth --demo ./videos/match_01_part.mp4

python demo.py tracking --load_model ../models/mot17_fulltrain.pth --demo ../videos/match_01_part.mp4
python demo.py tracking --load_model ../models/ctdet_coco_dla_2x.pth --demo ../videos/match_01_part.mp4

python demo.py tracking --load_model ../models/crowdhuman.pth --demo ../videos/match_01_part.mp4 

python demo.py tracking --load_model ../models/kitti_fulltrain.pth --dataset nuscenes --pre_hm --track_thresh 0.1 --demo ../videos/nuscenes_mini.mp4


# pose tracking
python demo.py tracking,multi_pose --load_model ../models/coco_pose_tracking.pth --demo ../videos/match_01_part.mp4 --track_thresh 0.1

python demo.py tracking,multi_pose --load_model ../models/coco_pose_tracking.pth --demo ../videos/Kei_stroke.mp4

<!-- python demo.py tracking,multi_pose --load_model ../models/coco_pose.pth --demo ../videos/match_01_part.mp4 -->

# train

bash ./experiments/coco_tracking.sh 



# colab
from google.colab import drive 
drive.mount('/content/drive')
%cd "/content/drive/My Drive/workspace/CenterTrack"

!pip install torchvision torch
!pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

CenterTrack_ROOT=/workspace/CenterTrack
pip install -r requirements.txt --user

%cd "src/lib/model/networks/"
!git clone https://github.com/CharlesShang/DCNv2/ # clone if it is not automatically downloaded by `--recursive`. -->
%cd "DCNv2"
!bash ./make.sh

%cd "/content/drive/My Drive/workspace/CenterTrack"
!bash ./experiments/coco_tracking.sh 

# mlflow
export PATH=$PATH:/home/ubuntu/.local/bin/
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
mlflow ui
