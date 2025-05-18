# Single UAV path planning based deep reinforcement learning

25年3月23日22:22留言，单无人机的路径规划代码已开源，视频讲解正在制作中...

如果您是新手小白，建议等视频教程推出后再看此代码呦~~~ 😃 单无人机路径规划视频教程的发布时间预计：即刻开始~1个月之间。

视频演示如下：


https://github.com/henbudidiao/UAV-path-planning/assets/64433060/42e9c7b4-6b6c-4e59-959e-f91b99687740

## 单无人机行为分析

图4-11以SAC算法训练的效果为例，展示了己方UAV在一个Monte Carlo测试回合中不同时刻的行为(其中，t<sub>1</sub> < t<sub>2</sub> < t<sub>3</sub> < t<sub>4</sub> < t<sub>5</sub> < t<sub>6</sub>)。
<div align=center>
  ![image](https://github.com/user-attachments/assets/3aad355d-9587-40f0-ac6d-80da9b014114)

<img src="https://github.com/user-attachments/assets/c38ad3fa-58a8-4503-a805-fea3541f5dc3" style="width:65%;height:65%;">
<img src="https://github.com/user-attachments/assets/6bf892d9-237b-44d5-b7d4-7062cce643cc" style="width:65%;height:65%;">

图4-11 单无人机不同时刻航路规划轨迹
</div>

如图4-11所示，己方UAV、敌机、目标地点和静态障碍物在t<sub>1</sub>时刻进行初始化生成；t<sub>2</sub>时刻己方UAV已经向目标地点移动了一些距离；t<sub>3</sub>时刻己方UAV与一架敌机有碰撞风险，此时己方UAV需要改变航向；t<sub>4</sub>时刻己方UAV开始改变航向进行避障；t<sub>5</sub>时刻己方UAV成功避开敌机；t<sub>6</sub>时刻己方UAV即将到达目标地点。显然，SAC算法训练后的无人机能够在动态不确定的仿真环境中进行自主航路规划，可以在安全避障和界内飞行的情形下到达目标地点。
