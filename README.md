# PointPillars Inference with TensorRT

This repository contains sources and model for [pointpillars](https://arxiv.org/abs/1812.05784) inference using TensorRT.
The model is created with [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and modified with onnx_graphsurgeon.

Overall inference has four phases:

- Voxelize points cloud into 10-channel features
- Run TensorRT engine to get detection feature
- Parse detection feature and apply NMS

## Model && Data

The demo use the velodyne data from KITTI Dataset.
The onnx file can be converted from [pre-trained model](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) with given script under "./tool".

### Prerequisites

We provide a [Dockerfile](docker/Dockerfile) to ease environment setup. Please execute the following command to build the docker image after nvidia-docker installation:
```
cd docker && docker build . -t pointpillar
```
We can then run the docker with the following command: 
```
nvidia-docker run --rm -ti -v /home/$USER/:/home/$USER/ --net=host --rm pointpillar:latest
```
For model exporting, please run the following command to clone pcdet repo and install custom CUDA extensions:
```
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet && git checkout 846cf3e && python3 setup.py develop
```
Download [PTM](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view) to ckpts/, then use below command to export ONNX model:
```
python3 tool/export_onnx.py --ckpt ckpts/pointpillar_7728.pth --out_dir model
```
Use below command to evaluate on kitti dataset:
```
sh tool/evaluate_kitti_val.sh
```

## Environments

- Nvidia Jetson Orin + CUDA 11.4 + cuDNN 8.9.0 + TensorRT 8.6.11

### Compile && Run

```shell
sudo apt-get install git-lfs && git lfs install
git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git
cd CUDA-PointPillars && mkdir build
. tool/environment.sh && cd build
cmake .. && make -j$(nproc)
cd ../ && sh tool/build_trt_engine.sh
cd build && ./pointpillar
```

#### Performance in FP16

Average perf in FP16 on the training set of KITTI dataset.

```
| Function(unit:ms) | Orin   |
| ----------------- | ------ |
| Voxelization      | 0.13   |
| Backbone & Head   | 5.99   |
| Decoder & NMS     | 1.64   |
| Overall           | 7.76   |
```

3D detection performance of moderate difficulty on the validation set of KITTI dataset.

```
|                   | Car@R11 | Pedestrian@R11 | Cyclist@R11  | 
| ----------------- | --------| -------------- | ------------ |
| CUDA-PointPillars | 77.01   | 52.08          | 62.57        |
| OpenPCDet         | 77.28   | 52.29          | 62.68        |
```

## Note

- Voxelization has random output since GPU processes all points simultaneously while points selection for a voxel is random.

## References

- [Detecting Objects in Point Clouds with NVIDIA CUDA-Pointpillars](https://developer.nvidia.com/blog/detecting-objects-in-point-clouds-with-cuda-pointpillars/)
- [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784)
