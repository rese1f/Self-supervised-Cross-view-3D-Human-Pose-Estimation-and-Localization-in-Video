# Dataexpand 1.9.7 Released

## Project Progress and Contribution Part

-   Wenhao Chai: 
    -   Rebulid the project and the argument inputs.
    -   Prepare to validate existing algorithms on new datasets.
    -   Proceed features on `cover.py`.
-   Jack Bai: 
    -   Helping Mu Xie with `camera.py`.
    -   Adding sequential collision handling to `collision.py`
-   Mu Xie:
    -   Completing features on `camera.py`.
-   Xinyu Jin:
    -   Writing paper.



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

## Pseudo-code For Collision Elimination Operations

![image-20210705130049360](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-05-050049.png)

![image-20210705130100821](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-07-05-050101.png)
