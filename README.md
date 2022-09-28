# Contents
- [Contents](#contents)
  - [Description](#description)
  - [Model Architecture](#model-architecture)
  - [Dataset](#dataset)
  - [Environment Requirements](#environment-requirements)
  - [Getting Start](#getting-start)
  - [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
    - [Evaluation Process](#evaluation-process)
  - [Citation](#citation)

## Description

![ARN_architecture](./src/pictures/ARN_model.png)

ARN builds on a C3D encoder for spatio-temporal video blocks to capture short-range action patterns. To improve training of the encoder,they introduce spatial and temporal self-supervision by rotations, and spatial and temporal jigsaws and propose "attention by alignment", a new data splits for a systematic comparison of few-shot action recognition algorithms.

## Dataset

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

## Environment Requirements

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

## Getting Start

- Run on GPU

```text
cd scripts/

# run training example
bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]

# run distributed training example
bash train_distribute.sh [PROJECT_PATH] [DATA_PATH]

# run evaluation example
bash eval_standalone.sh [PROJECT_PATH] [DATA_PATH]
```

## Script Description

### Script and Sample Code

```text
.
│  infer.py                                     // infer script
│  README.md                                    // descriptions about arn
│  train.py                                     // training script
│
├─scripts
│      eval_standalone.sh                       // shell script for testing on GPU
│      train_distribute.sh                      // shell script for distributed training on GPU
│      train_standalone.sh                      // shell script for training on GPU
│
└─src
    │
    ├─config/arn
    │      arn.yaml                           // ARN parameter configuration
    │
    ├─data
    │  │  builder.py                            // build data
    │  │  download.py                           // download dataset
    │  │  generator.py                          // generate video dataset
    │  │  images.py                             // process image
    │  │  kinetics400.py                        // kinetics400 dataset
    │  │  ucf101.py                             // UCF101 dataset
    │  │  meta.py                               // public API for dataset
    │  │  path.py                               // IO path
    │  │  ucf101.py                             // ucf101 dataset
    │  │  video_dataset.py                      // video dataset
    │  │
    │  └─transforms
    │          builder.py                       // build transforms
    │          video_center_crop.py             // center crop
    │          video_normalize.py               // normalize
    │          video_random_crop.py             // random crop
    │          video_random_horizontal_flip.py  // random horizontal flip
    │          video_reorder.py                 // reorder
    │          video_rescale.py                 // rescale
    │          video_reshape.py                 // reshape
    │          video_resize.py                  // resize
    │          video_short_edge_resize.py       // short edge resize
    │
    ├─example/arn
    │      arn_ucf101_eval.py              		// eval arn model
    │      arn_ucf101_train.py             		// train arn model
    │      2001.03905.pdf                  		// arn paper
    |
    ├─loss
    │      builder.py                           // build loss
    │
    ├─models
    │  │  builder.py                            // build model
    │  │  arn.py                                // arn model
    │  │
    │  └─layers
    │          adaptiveavgpool3d.py             // adaptive average pooling 3D.
    │          avgpool3d.py                     // average pooling 3D
    │          dropout_dense.py                 // dense head
    │          inflate_conv3d.py                // inflate conv3d block
    │          resnet3d.py                      // resnet backbone
    │          squeeze_excite3d.py              // squeeze and excitation
    │          swish.py                         // swish activation function
    │          unit3d.py                        // unit3d module
    │
    ├─optim
    │      builder.py                           // build optimizer
    │
    ├─schedule
    │      builder.py                           // build learning rate shcedule
    │      lr_schedule.py                       // learning rate shcedule
    │
    └─utils
            callbacks.py                        // eval loss monitor
            check_param.py                      // check parameters
            class_factory.py                    // class register
            config.py                           // parameter configuration
            six_padding.py                      // convert padding list into tuple

```

### Script Parameters

- config for arn 
  
```text
# model architecture
model_name: arn

# The dataset sink mode.
dataset_sink_mode: False

# Context settings.
context:
    mode: 0 #0--Graph Mode; 1--Pynative Mode
    device_target: "GPU"

# model settings of every parts
model:
    type: arn
    support_num_per_class: 5
    query_num_per_class: 3
    class_num: 5
    is_c3d: False
    in_channels: 3
    out_channels: 64
    jigsaw: 10
    sigma: 100

# learning rate for training process
learning_rate:
    lr: 0.001

# optimizer for training process
optimizer:
    type: 'Adam'

loss:
    type: MSELoss

train:
    pre_trained: True
    pretrained_model: "/home/huyt/807_ARN_ucf_CROSS0.7446666666666667.ckpt"
    ckpt_path: "./output/"
    epochs: 100
    save_checkpoint_epochs: 5
    save_checkpoint_steps: 1875
    keep_checkpoint_max: 10
    run_distribute: True

infer:
    pretrained_model: "/home/huyt/807_ARN_ucf_CROSS0.7446666666666667.ckpt"

# UCF101 dataset config
data_loader:
    train:
        dataset:
              type: UCF101
              path: "/home/publicfile/UCF101"
              batch_size: 1
              split: 'train'
              shuffle: False
              seq: 16
              num_parallel_workers: 1
              suffix: "task"
              task_num: 100000
              task_n: 5
              task_k: 1
              # task_k: 5
              task_q: 1
        map:
            operations:
                - type: VideoReshape
                  shape: [-1, 240, 320, 3]
                - type: VideoResize
                  size: [128, 128]
                - type: VideoToTensor
                - type: VideoNormalize
                  mean: [0.3474, 0.3474, 0.3474]
                  std: [0.2100, 0.2100, 0.2100]
                - type: VideoReshape
                  shape: [3, -1, 16, 128, 128]
            input_columns: ["video"]

    eval:
        dataset:
            type: UCF101
              path: "/home/publicfile/UCF101"
              batch_size: 1
              split: 'test'
              shuffle: False
              seq: 16
              num_parallel_workers: 1
              suffix: "task"
              task_num: 100
              task_n: 5
              task_k: 1
              # task_k: 5
              task_q: 1
        map:
            operations:
                - type: VideoReshape
                  shape: [-1, 240, 320, 3]
                - type: VideoResize
                  size: [128, 128]
                - type: VideoToTensor
                - type: VideoNormalize
                  mean: [0.3474, 0.3474, 0.3474]
                  std: [0.2100, 0.2100, 0.2100]
                - type: VideoReshape
                  shape: [3, -1, 16, 128, 128]             
            input_columns: ["video"]
    group_size: 1
```

## Citation

If you find this project useful in your research, please consider citing:

```latex
@misc{arn_misdspore,
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