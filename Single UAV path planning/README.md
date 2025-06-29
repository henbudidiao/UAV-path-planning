# Single UAV path planning based deep reinforcement learning

25年6月29日，上集视频教程已出，路径：https://www.bilibili.com/video/BV1jkgZzKEe1/?spm_id_from=333.1387.homepage.video_card.click&vd_source=b94eb1c3a80dbcc148ebec37b6e5ff87

下集视频讲解教程正在制作中...预计时间：一周到一年之间。 。 。 。

视频演示如下：

https://github.com/henbudidiao/UAV-path-planning/assets/64433060/42e9c7b4-6b6c-4e59-959e-f91b99687740

## 单无人机行为分析

图4-11以SAC算法训练的效果为例，展示了己方UAV在一个Monte Carlo测试回合中不同时刻的行为(其中，t<sub>1</sub> < t<sub>2</sub> < t<sub>3</sub> < t<sub>4</sub> < t<sub>5</sub> < t<sub>6</sub>)。
<div align=center>
<img src="https://github.com/user-attachments/assets/3aad355d-9587-40f0-ac6d-80da9b014114" style="width:64%;height:64%;">
<img src="https://github.com/user-attachments/assets/eab6fe98-1377-42a0-8c7c-4a7c3c223557" style="width:65%;height:65%;">

图4-11 单无人机不同时刻航路规划轨迹
</div>

如图4-11所示，己方UAV、敌机、目标地点和静态障碍物在t<sub>1</sub>时刻进行初始化生成；t<sub>2</sub>时刻己方UAV已经向目标地点移动了一些距离；t<sub>3</sub>时刻己方UAV与一架敌机有碰撞风险，此时己方UAV需要改变航向；t<sub>4</sub>时刻己方UAV开始改变航向进行避障；t<sub>5</sub>时刻己方UAV成功避开敌机；t<sub>6</sub>时刻己方UAV即将到达目标地点。显然，SAC算法训练后的无人机能够在动态不确定的仿真环境中进行自主航路规划，可以在安全避障和界内飞行的情形下到达目标地点。
