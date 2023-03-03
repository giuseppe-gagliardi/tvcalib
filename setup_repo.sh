#!/bin/bash

# checkout submodule
cd  sn_segmentation
git submodule update --init --recursive
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
pip3 install gdown
gdown 1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM  #see here: https://github.com/SoccerNet/sn-calibration#install
cd ..


mkdir -p data/segment_localization
wget https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt -O data/segment_localization/train_59.pt

mkdir -p data/datasets/calibration

python3 download_dataset.py 

cd data/datasets/calibration

unzip valid.zip
unzip test.zip
unzip train.zip

cd ../../../

wget https://tib.eu/cloud/s/483Bqf78dDMcx2H/download/test_match_info_cam_gt.json -O data/datasets/calibration/test/match_info_cam_gt.json
wget https://tib.eu/cloud/s/WdSqM3WbyKQ36pm/download/val_match_info_cam_gt.json -O data/datasets/calibration/valid/match_info_cam_gt.json

# move annotation file to respective dataset directory
mv data/datasets/calibration/valid data/datasets/sncalib-valid
mv data/datasets/calibration/test data/datasets/sncalib-test
mv data/datasets/calibration/train data/datasets/sncalib-train

mkdir -p data/datasets/wc14-test
cd data/datasets/wc14-test/

# Images and provided homography matrices from test split
wget https://nhoma.github.io/data/soccer_data.tar.gz
tar -zxvf soccer_data.tar.gz raw/test --strip-components 2
# Our additional segment annotations
wget https://tib.eu/cloud/s/Jz4x2KsjinEEkwQ/download/wc14-test-additional_annotations_wacv23_theiner.tar -O wc14-test-additional_annotations_wacv23_theiner.tar
tar xvf wc14-test-additional_annotations_wacv23_theiner.tar

cd ../../../

pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio
