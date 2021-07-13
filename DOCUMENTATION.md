# Documentation

- [Documentation](#documentation)
  * [Prepare Dataset](#prepare-dataset)
    + [Principles](#principles)
    + [Execution](#execution)
  * [Data Enhancement](#data-enhancement)
    + [Principles](#principles-1)
    + [Execution](#execution-1)
  * [Video Pose 3D](#video-pose-3d)
  * [STAR Model](#star-model)

This guide includes **all** important information about the whole project and includes the essential principles of the code logic. Please be **sure** that you’ve read [README.md](./README.md) thoroughly before you move on.

The following part of this file defines all the used classes/functions in each block. Again, please turn back to [README.md](./README.md) if you have not understood the macro functionalities of all the workflow parts.

## Prepare Dataset

### Principles

You can find the file [prepare_dataset.py](./DataGenerators/prepare_dataset.py) in the dir `DataGenerators`. This part simply combine `n` data files into one into the dir `DataGenerators/data`. With the generated combined dataset we’re able to improve the effeciency of IO for the machine whether it’s on Windows, Mac or Linux - for over 50% speed-up. The possible explanation of this part is that, the newly formed `data_3d_h36m.npz` is stored in the working environment (instead of the disk) and is much easier to retrieve by the machine.

By graph presentation, the algorithm of the data preparation part can be expressed as below.

![PrepareDatasetAlgo](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-110447.png)

### Parameters

#### `extract(dict_keys, min, max)`

`dict_keys`: 

`min`:

`max`:

#### `pre_process(array, shift, distance, rotation)`

`array`:

`shift`:

`distance`:

`rotation`:

### Execution

For execution, simply type

```shell
cd ./DataGenerators/
python3 prepare_dataset.py
```

If you’re on the right way, you will find `data_3d_h36m.npz` in dir `DataGenerators/data`. 

## Data Enhancement

### Principles

This part is core of this project, and will be explained detailedly. The overall algorithm of this part can be expressed as a self-developed algorithm, as shown below.

![image-20210712191740717](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-111740.png)

Each part of the algorithm is separated from each other and is implemented individually.

-   Random Functioning. The random functioning part comprises of the random translation part and the random rotation part. Both parts take one person and do ramdom switches on the person, after which multiple persons will be combined and put into one coordinate system and form a new series of data. The dataset is thus “enhanced” to a new one, with multiple persons and a double/triple complexity.
-   Collision Elimination. This part deals with the collision after the random functioning part. After the random functioning part the persons are likely to collide with each other, which is forbidden in real datasets. Our group thus provides a self-developed algorithm to the dataset to gain a *shift vector* for each person to avoid collisions.
-   Camera Generation. After the collision elimination part we now gain the 3 dimensional dataset with multiple persons in each frame. Now by the extrinsic camera matrix we gain the corresponding 2 dimensional dataset, and by the intrinsic camera matrix we gain the corresponding 2 dimensional dataset without deleterious distortions. This part also simulates different kinds of motions of cameras, e.g. phone camera and surveillance camera.

By graph presentation, the algorithm of data enhancement part can be expressed as below.

![DataEnhancementAlgo](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-110456.png)

Now we look into each part one by one.

#### Random Functioning

This part basically deal with the movements of the raw data. 

#### Collision Elimination

The collision elimination part is a brand new, self-developed algorithm, *Sequential Approach for Eliminating Individual Collisions*.  The pseudo code is shown below.

![image-20210705130049360](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-112522.png)

In this function, there is another complicated function `find_shift_vector()`, which is illustrated below.

![image-20210705130100821](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-112610.png)

#### Camera generator

This part basically 



### Execution/Super-parameters

For execution simply type

```shell
cd ./DataGenerators/
python3 run.py
```

If you’re on the right way, you will find `data_multi_3d_h36m.npz` in dir `DataGenerators/output`.

If you want to changes the super-parameters, you have these choices below:

| Arg.   | Abbr.           | Meaning                                                      | Default   |
| ------ | --------------- | ------------------------------------------------------------ | --------- |
| `-d`   | `--dataset`     | The dataset to be expanded.                                  | *h36m*    |
| `-min` | `--min`         | The minimum number of persons in the new dataset.            | 2         |
| `-max` | `--max`         | The maximum number of persons in the new dataset             | 4         |
| `-s`   | `--shift`       | The mean value for shifting the timeline.                    | 500       |
| `-t`   | `--translation` | The mean value for the random translation part.              | 1000      |
| `-r`   | `--rotation`    | Whether rotating a single raw data (*True*) or not (*False*). | *True*    |
| `-n`   | `--number`      | The number of the generated datasets (all at once).          | 16        |
| `-c`   | `camera`        | The type of camera used to generate the new dataset.         | [‘Phone’] |
| `-v`   | `--view`        | The number of view sites.                                    | 1         |

If you are curious about the source code implementing the parameters, turn to [DataGenerators/arguments.py](./DataGenerators/arguments.py)

## Video Pose 3D

### Principles

Note that although this part is already implemented, it’s not reported yet because it containes much mathematical knowledge. The detailed illustration of this part will be shown in the **essay**.

This workflow part aims to switch 2D pose to 3D. The main technique used is the regressor, with a back-propagation process `back_forward()` to gain the feedback system with a loss value.

The basic logic of this part can be expressed below.

![VideoPose3DLayout](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-13-070950.png)

### Execution/Super-parameters

| Arg.    | Abbr.                      | Meaning                                                      | Default                   |
| ------- | -------------------------- | ------------------------------------------------------------ | ------------------------- |
| `-d`    | `--dataset`                | The dataset to be expanded.                                  | *h36m*                    |
| `-k`    | `--keypoints_number`       | The number of key points in each frame.                      | 17                        |
| `-c`    | `--checkpoint`             | The dir for outputting checkpoints.                          | `checkpoint`              |
| `-l`    | `--load`                   | The checkpoint file to be loaded.                            | `pretrained_h36m_cpn.bin` |
| /       | `--save`                   | The checkpoint file to be saved.                             | `trained_h36m_cpn.bin`    |
| `-v`    | `--multi-view`             | Whether the data set has multi-view (*True*) or not (*False*). | *False*                   |
| `-eval` | `--evaluate`               | Whether the machine makes evaluation after getting the 3D ground truth (*True*) or not. (*False*) | *True*                    |
| `-u`    | `--update`                 | Whether the machine updates the parameters of the model (*True*) or not (*False*). | *True*                    |
| `-o`    | `--output`                 | Whether the output predicts the 3D pose (*True*) or not(*False*). | *False*                   |
| /       | `--export-training-curves` | If flagged, save training curves (as `*.png` files)          | **FLAG HAS NO DEFAULT**   |
| `-s`    | `--stride`                 | The trunk size to use when training.                         | 1                         |
| `-e`    | `--epochs`                 | The number of training epochs.                               | 4                         |
| `-drop` | `--dropout`                | The probability for dropouts.                                | 0.25                      |
| `-lrd`  | `--lr-decay`               | The decay of learning rate through each epoch.               | 0.95                      |
| `-arc`  | `--architecture`           | The filter widths (separated by commas).                     | 3, 3, 3, 3, 3             |
| /       | `--causal`                 | If flagged, use causal convolutions for real-time processing. | **FLAG HAS NO DEFAULT**   |
| `-ch`   | `--channels`               | The number of channels in the convolution layers.            | 1024                      |

If you are curious about the source code implementing the parameters, turn to [VideoPose3D/common/arguments.py](VideoPose3D/common/arguments.py).

## STAR Model

The vertex layout of STAR model is shown below.

![WechatIMG572](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-113623.png)

This part is still under developement. Do **not** edit this part until version 2.0.0 is released.