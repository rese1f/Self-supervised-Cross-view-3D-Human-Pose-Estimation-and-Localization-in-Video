# Documentation
This guide includes **all** important information about the whole project and includes the essential principles of the code logic. Please be **sure** that you’ve read `README.md` thoroughly before you move on.

## Codeblock Level Logic

The codeblock level logic of this project is conposed of four blocks:

-   Workflow Block. The workflow block defines the most readable workflow of the whole project, and it is embodied by the directories of the project.
-   Trigger Block. The trigger block is the trigger Python script of **each** workflow block. In other words, each workflow block contains one and only one trigger block. For example, `run.py` is the trigger block of the workflow dir `DataGenerators`.
-   Visualization Block. The visualization block is elaborated to give a feedback the data processing. Developers are required to utilize the visualization blocks to examine whether the current workflow block is behaving malfunctionally. 
-   Library Block. The library blocks include all class/function definitions for the corresponding workflow block. This block often consists of a large amount of files, each in charge of a certain & separate library of function definitions.

By graph presentation, the codeblock level logic can be expressed as

![BlockLogic](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-095640.png)

## Overall Layout of the Project

The overall layout of the Project is composed of 4 workflow blocks hitherto, each behaving on a certain functionality.

-   Prepare Dataset. In this part, we transform pieces of raw data from `Human3.6M` Dataset into one Python-readable piece of data, in order to enhance the speed of input/output of the data with the machine. This step is inevitable and has been proved to improve at least 50% efficiency of the program.
-   Data Enhancement. In this part, we utilize the transfered dataset to make a new dataset with multiple persons in one frame. The basic idea is, in each frame we construct 3 persons into the same scene to enhance the dataset.
-   Video Pose 3D. This part is designed to 

![OverallLayout](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-12-095701.png)

