# Documentation
This guide includes **all** important information about the whole project and includes the essential principles of the code logic. Please be **sure** that you’ve read [README.md](./README.md) thoroughly before you move on.

The following part of this file defines all the used classes/functions in each block. Again, please turn back to [README.md](./README.md) if you have not understood the macro functionalities of all the workflow parts.

## Prepare Dataset

You can find the file [prepare_dataset.py](./DataGenerators/prepare_dataset.py) in the dir `DataGenerators`. This part simply combine `n` data files into one into the dir `DataGenerators/data`. With the generated combined dataset we’re able to improve the effeciency of IO for the machine whether it’s on Windows, Mac or Linux - for over 50% speed-up. The possible explanation of this part is that, the newly formed `data_3d_h36m.npz` is stored in the working environment (instead of the disk) and is much easier to retrieve by the machine.

For execution, simply type

```shell
cd ./DataGenerators/
python3 prepare_dataset.py
```

If you’re on the right way, you will find `data_3d_h36m.npz` in dir `DataGenerators/data`. 

## Data Enhancement

This part is core of this project, and will be explained detailedly. The overall algorithm of this part can be expressed as a self-developed algorithm, as shown below.



By graph presentation, the algorithm of data enhancement part can be expressed as below.

![DataEnhancementAlgo](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-103901.png)

## Video Pose 3D







## STAR Model

