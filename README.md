# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.

Overall inference has below phases:

- Voxelize points cloud into 10-channel features
- Run TensorRT engine to get detection feature
- Parse detection feature and apply NMS

## Prerequisites

### Prepare Model && Data

We provide a [Dockerfile](docker/Dockerfile) to ease environment setup. Please execute the following command to build the docker image after nvidia-docker installation:
```
cd docker && docker build . -t pointpillar-new
```
Come back to the root directory:
```
cd ..
```
We can then run the docker with the following command: 
```
docker run --rm -ti --gpus all -v "%cd%":/home/working_dir --net=host pointpillar-new:latest
```
Navigate to the respective directory within our docker image:
```
cd home/working_dir
```
For model exporting, please run the following command to clone pcdet repo and install custom CUDA extensions:
```
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout pointpillar-fix && python3 setup.py develop
```
Create config files for the kitti dataset while in OpenPCDet directory:
```
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos ../cfgs/dataset_configs/kitti_dataset.yaml --data_path ../data/kitti_new
```
Go back to the root directory within docker (./home/working_dir)
```
cd ..
```
Use below command to train ONNX model to given kitti dataset using the checkpoint in ./ckpts:
```
python3 tool/train_model.py --cfg_file cfgs/kitti_models/pointpillar.yaml --data_path ./data/kitti_new --ckpt ./ckpts/pointpillar_7728.pth --out_dir ./output --epochs 10 --batch_size 4
```

