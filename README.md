# arn_mindspore
- [Description](https://github.com/ZJUT-ERCISS/fairmot_mindspore#description)
- [Model Architecture](https://github.com/ZJUT-ERCISS/fairmot_mindspore#model-architecture)
- [Dataset](https://github.com/ZJUT-ERCISS/fairmot_mindspore#dataset)
- [Environment Requirements](https://github.com/ZJUT-ERCISS/fairmot_mindspore#environment-requirements)
- Quick Start
  - [Requirements Installation](https://github.com/ZJUT-ERCISS/fairmot_mindspore#requirements-installation)
  - [Dataset Preparation](https://github.com/ZJUT-ERCISS/fairmot_mindspore#dataset-preparation)
  - [Model Checkpoints](https://github.com/ZJUT-ERCISS/fairmot_mindspore#model-checkpoints)
  - [Running](https://github.com/ZJUT-ERCISS/fairmot_mindspore#running)
- Script Description
  - Training Process
    - [Training](https://github.com/ZJUT-ERCISS/fairmot_mindspore#training)
    - [Distributed Training](https://github.com/ZJUT-ERCISS/fairmot_mindspore#distributed-training)
  - [Evaluation Process](https://github.com/ZJUT-ERCISS/fairmot_mindspore#evaluation-process)
- Model Description
  - [Performance](https://github.com/ZJUT-ERCISS/fairmot_mindspore#performance)
- [Citation](#Citation)

# Description

ARN builds on a C3D encoder for spatio-temporal video blocks to capture short-range action patterns. To improve training of the encoder,they introduce spatial and temporal self-supervision by rotations, and spatial and temporal jigsaws and propose "attention by alignment", a new data splits for a systematic comparison of few-shot action recognition algorithms.

# Model Architecture

![ARN_architecture](./src/pictures/ARN_model.png)

The overall network architecture of ARN is shown below:

[link](#[[2001.03905\] Few-shot Action Recognition with Permutation-invariant Attention (arxiv.org)](https://arxiv.org/abs/2001.03905))

# Dataset

Dataset used: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

- Description: UCF101 is an action recognition data set of realistic action videos, collected from YouTube, having 101 action categories. This data set is an extension of UCF50 data set which has 50 action categories.

- Dataset size：13320 videos
    - Note：Use the official Train/Test Splits([UCF101TrainTestSplits](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)).
- Data format：rar
    - Note：Data will be processed in dataset_preprocess.py
- Data Content Structure

```text
.
└─ucf101                                    // contains 101 file folder
  ├── ApplyEyeMakeup                        // contains 145 videos
  │   ├── v_ApplyEyeMakeup_g01_c01.avi      // video file
  │   ├── v_ApplyEyeMakeup_g01_c02.avi      // video file
  │    ...
  ├── ApplyLipstick                         // contains 114 image files
  │   ├── v_ApplyLipstick_g01_c01.avi       // video file
  │   ├── v_ApplyLipstick_g01_c02.avi       // video file
  │    ...
  ├── ucfTrainTestlist                      // contains category files
  │   ├── classInd.txt                      // Category file.
  │   ├── testlist01.txt                    // split file
  │   ├── trainlist01.txt                   // split file
  ...
```

# Environment Requirements

- Framework
  - [MindSpore](https://www.mindspore.cn/install/en)

- Requirements

```text
Python and dependencies
    - python 3.7.5
    - decord 0.6.0
    - imageio 2.21.1
    - imageio-ffmpeg 0.4.7
    - mindspore-gpu 1.6.1
    - ml-collections 0.1.1
    - matplotlib 3.4.1
    - numpy 1.21.5
    - Pillow 9.0.1
    - PyYAML 6.0
    - scikit-learn 1.0.2
    - scipy 1.7.3
    - pycocotools 2.0
```

- For more information, please check the resources below：
  - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick Start

## Requirements Installation

```text
pip install -r requirements.txt
```

## Dataset Preparation

ARN model uses UCF101 dataset to train and validate in this repository.Please refer to their website ([UCF101](https://www.crcv.ucf.edu/data/UCF101.php))to download and prepare all the data.

**Configure path to dataset root** in `data/data.json` file.

## Model Checkpoints

The pretrain model is trained on the the UCF101 for 30 epochs. It can be downloaded here:

## Running

- Run on GPU

```text
cd scripts/

# run training example
bash run_standalone_train.sh [PROJECT_PATH] [DATA_PATH]

# run distributed training example
bash run_distribute_train.sh [PROJECT_PATH] [DATA_PATH]

# run evaluation example
bash run_standalone_eval.sh [PROJECT_PATH] [DATA_PATH]
```

# Script Description

## Training Process

### Training Alone

Run `scripts/run_standalone_train.sh` to train the model standalone. The usage of the script is:

#### Running on GPU

```
bash scripts/run_standalone_train.sh [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the training procedure:

```
bash scripts/run_standalone_train.sh ./arn.yaml ./ARN_ucf_MSE.ckpt
```

The model checkpoint will be saved into `./output`.

### Distributed Training

Run `scripts/run_distribute_train.sh` to train the model distributed. The usage of the script is:

#### Running on GPU

```
bash scripts/run_distribute_train.sh [DEVICE_NUM] [VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [config_file] [pretrained_model]
```

For example, you can run the shell command below to launch the distributed training procedure:

```
bash scripts/run_distribute_train.sh 8 0,1,2,3,4,5,6,7 ./arn.yaml ./ARN_ucf_MSE.ckpt
```

The above shell script will run distribute training in the background. You can view the results through the file `train/tran.log`.

The model checkpoint will be saved into `train/ckpt`.

## Evaluation Process

The evaluation data set was [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

Run `scripts/run_eval.sh` to evaluate the model. The usage of the script is:

```
bash scripts/run_standalone_eval.sh [device] [config] [load_ckpt] [dataset_dir]
```

For example, you can run the shell command below to launch the validation procedure.

```
bash scripts/run_standalone_eval.sh GPU ./arn.yaml ./ARN_ucf_MSE.ckpt data_path
```

The eval results can be viewed in `eval/eval.log`.

# [Model Description](https://github.com/ZJUT-ERCISS/fairmot_mindspore#contents)

## [Performance](https://github.com/ZJUT-ERCISS/fairmot_mindspore#contents)

### ARN on UCF101 dataset with detector

#### Performance parameters

| Parameters          | GPU Standalone              | GPU Distributed             |
| ------------------- | --------------------------- | --------------------------- |
| Model Version       | ARN                         | ARN                         |
| Resource            | RTX 3090 24GB               | 8x RTX 3090 24GB            |
| Uploaded Date       | 25/06/2021 (day/month/year) | 21/02/2021 (day/month/year) |
| MindSpore Version   | 1.2.0                       | 1.5.0                       |
| Training Dataset    | UCF101                      | UCF101                      |
| Evaluation Dataset  | UCF101                      | UCF101                      |
| Training Parameters | epoch=30, batch_size=4      | epoch=30, batch_size=12     |
| Optimizer           | Adam                        | Adam                        |
| Loss Function       | MSELoss                     | MSELoss                     |
| Train Performance   |                             |                             |

# Citation

If you find this project useful in your research, please consider citing:

```text
@article{DBLP:journals/corr/abs-2001-03905,
  author    = {Hongguang Zhang and
               Li Zhang and
               Xiaojuan Qi and
               Hongdong Li and
               Philip H. S. Torr and
               Piotr Koniusz},
  title     = {Few-shot Action Recognition via Improved Attention with Self-supervision},
  journal   = {CoRR},
  volume    = {abs/2001.03905},
  year      = {2020}
}
```



```latex
@misc{arn_mindspore,
    author = {Zhang, Hongguang and Zhang, Li and Qi, Xiaojuan and Li, Hongdong and Torr, Philip HS
                and Koniusz, Piotr},
    title = {Mindspore Video Models},
    year = {2022},
    publisher = {GitHub},
    journal = {GitHub repository},
    doi = {10.1007/978-3-030-58558-7_31},
    howpublished = {\url{https://github.com/ZJUT-ERCISS/arn_misdspore}}
}
```