# Dataexpand 1.9.7 Released

## Codeblock Level Logic

The codeblock level logic of this project is conposed of four blocks:

-   Workflow Block. The workflow block defines the most readable workflow of the whole project, and it is embodied by the directories of the project.
-   Trigger Block. The trigger block is the trigger Python script of **each** workflow block. In other words, each workflow block contains one and only one trigger block. For example, `run.py` is the trigger block of the workflow dir `DataGenerators`.
-   Visualization Block. The visualization block is elaborated to give a feedback the data processing. Developers are required to utilize the visualization blocks to examine whether the current workflow block is behaving malfunctionally. 
-   Library Block. The library blocks include all class/function definitions for the corresponding workflow block. This block often consists of a large amount of files, each in charge of a certain & separate library of function definitions.

By graph presentation, the codeblock level logic can be expressed as below.

![BlockLogic](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-101229.png)

## Overall Layout of the Project

The overall layout of the Project is composed of 4 workflow blocks hitherto, each behaving on a certain functionality.

-   Prepare Dataset. In this part, we transform pieces of raw data from `Human3.6M` Dataset into one Python-readable piece of data, in order to enhance the speed of input/output of the data with the machine. This step is inevitable and has been proved to improve at least 50% efficiency of the program.
-   Data Enhancement. In this part, we utilize the transfered dataset to make a new dataset with multiple persons in one frame. The basic idea is, in each frame we construct 3 persons into the same scene to enhance the dataset.
-   Video Pose 3D. This part is designed to transform the data with 2D Pose to 3D Pose. The substantive technique used in this part is regression, and we’ve designed an elaborated visualization part for this workflow block. The main model of this part is retrieved from the Github Project `VideoPose3D`.
-   STAR Model. This model is not yet elaborated and will be worked on after the release of version 2.0.0.

By graph presentation, the overall layout of the project can be expressed as below.

![OverallLayout](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-101231.png)

For dissecting all the parts please turn to [DOCUMENTATION.d](./DOCUMENTATION.md), which elaborates on the explanation of all functions and parameters passed in for each code block.

## Algorithm Flowchart

This is the flowchart of our algorithm about operations on *Human3.6M*. The infeed dataset may be expanded to datasets other than *Human3.6M*.

![2021-06-14-144814](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-06-15-021149.png)

## Vertex Layout of *Human3.6M* Dataset

### Graphical Layout (Left *Complete*, Right *Simple*)

<center class="half">    <img src="http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-06-17-082747.png" width="300"/><img src="http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-06-30-091119.png" width="450"/></center>

### Mapping Between The Two Sets

![image-20210630171302068](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-06-30-091302.png)

Please refer to [this website](https://www.stubbornhuang.com/529/) for more details.

## Vertex Layout of the *STAR* Model

![WechatIMG572](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-06-30-091732.png)

## Logic of VideoPose3D Project

![image-20210630171845523](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-06-30-091845.png)

## A More Accurate Logic

![image-20210705212050018](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-05-132050.png)



## Pseudo-code For Collision Elimination Operations

![image-20210705130049360](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-05-050049.png)

![image-20210705130100821](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-05-050101.png)
