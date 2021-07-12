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

-   Random Functioning. The random functioning part comprises of 
-   Collision Elimination.
-   Camera Generation.

By graph presentation, the algorithm of data enhancement part can be expressed as below.

![DataEnhancementAlgo](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-110456.png)

Now we look into each part one by one.

#### Random Functioning



#### Collision Elimination

The collision elimination part is a brand new, self-developed algorithm, *Sequential Approach for Eliminating Individual Collisions*.  The pseudo code is shown below.

![image-20210705130049360](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-112522.png)

In this function, there is another complicated function `find_shift_vector()`, which is illustrated below.

![image-20210705130100821](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-112610.png)

#### Camera generator





### Execution

For execution simply type

```shell
cd ./DataGenerators/
python3 run.py
```

If you’re on the right way, you will find `data_multi_3d_h36m.npz` in dir `DataGenerators/output`. 

## Video Pose 3D

![image-20210630171845523](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-113712.png)

![image-20210705212050018](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-113719.png)



## STAR Model

The vertex layout of STAR model is shown below.

![WechatIMG572](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-113623.png)

This part is still under developement. Do **not** edit this part until version 2.0.0 is released.