# 一份简洁的程序使用说明书

## 1 软件结构
《基于最大熵安全强化学习的无人机路径规划》的程序主要包含以下几个模块：文字信息模块——`info.py`、常数变量模块——`constants.py`、无人机创建模块——`player.py`、声音图像导入模块——`tools.py`强化学习环境模块——`path_env.py`、主程序模块——`main_SAC.py`等。

每一个模块都对应一个文件，打开后的架构如下图所示：
<div align=center>
<img src="https://github.com/user-attachments/assets/82bbf281-a689-48cb-b2b4-40fb0ad32686">

图1 软件结构界面
</div>

## 2 程序启动与运行说明

首先，注意打开文件的层级，并确保本地电脑拥有软件所需的素材包（即图片与音效文件）。然后，修改图片加载的文件路径，以下图为例，我的图片存放路径在G盘的image文件夹中。
<div align=center>
<img src="https://github.com/user-attachments/assets/2d8ccc43-b375-467b-b9f1-d9fa8ddaf649">
  
 图2 图片加载路径
</div>

此外，在确保安装好所有所需的python依赖包后，**pygame版本: 2.1.2; gym 版本: 0.19.0; pytorch 版本: 1.10.0+cu113; numpy: 1.23.1; pickle这个包现在好像是下不到了，用pickle4或pickle5顶替试试，如果还是不行，也无所谓，直接删掉所有用到pickle的地方，无非就是plot文件的图出不来而已。**（版本很重要，尤其是新手小白，解决报错的能力较弱，一定要和我保持一致）直接运行`main_SAC.py`文件即可正常启动程序。其中，可以自行更改训练好的神经网络权重参数存放路径，如下图：
<div align=center>
<img src="https://github.com/user-attachments/assets/857bb7e3-437b-476e-830f-798abffae637">
 
 图3 神经网络权重参数存放路径
</div>

## 3 程序使用说明

代码用法：先把Switch标志位赋为0，先训练。训练时，建议将RENDER标志位赋为False，关闭可视化可以让训练时间大大缩短。等待训练结束后画出奖励随回合数的曲线图形，神经网络的参数会被保存在文件夹中。然后，把Switch标志为赋为1进行路径规划的测试。测试时，将RENDER标志位赋为True即可打开可视化，可以看到无人机在空战环境中的运动情况。

在运行了`main_SAC.py`文件后，电脑桌面会弹出软件界面。如图4所示：
<div align=center>
<img src="https://github.com/user-attachments/assets/9af702fe-5c1a-4f85-a7bd-b6a2c9327e25">

 图4 无人机路径规划界面（新版界面）
</div>

`plot.py`文件中存放了所有的画图代码，文件中有代码注解。注释的代码请自行解注，以根据自己的需要生成。

***不久之后我会出一个视频教程，大伙可以github代码配合视频讲解一并服用，包药到病除，效果更佳哦~~ :smiley:***

25年6月29日，视频已出（上集），地址请见：https://www.bilibili.com/video/BV1jkgZzKEe1/?spm_id_from=333.1387.homepage.video_card.click&vd_source=b94eb1c3a80dbcc148ebec37b6e5ff87
