# 程序使用说明

## 1 软件结构

《基于 MASAC 强化学习算法的多无人机协同路径规划》的程序主要包含以下几个模块：文字信息模块——`info.py`、常数变量模块——`constants.py`、无人机创建模块——`player.py`、声音图像导入模块——`tools.py`强化学习环境模块——`path_env.py`、主程序模块——`main_SAC.py`等。

每一个模块都对应一个文件，各个文件之间的架构：首先创建一个env的文件夹，该env文件里再创建assignment文件夹、rl_env的python包和主程序模块。assignment文件夹里再创建components文件夹、source文件夹、声音图像导入模块`tools.py`和常数变量模块`constants.py`。components文件夹里存放文字信息模块`info.py`和无人机创建模块`player.py`；source文件夹是一个素材包，其中的music文件夹用来存放音乐，image文件夹用来存放图片； rl_env包里存放强化学习环境模块`path_env.py`。创建好后的架构如下图所示：
![image](https://github.com/user-attachments/assets/091ed2b4-6aa8-4fe9-931d-69a4d1860f78)

<p align="center">
 xx
</p>!


