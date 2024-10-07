# Multi-UAVs path planning based multi-agent deep reinforcement learning

话不多说，作者我开始这项工作的目的可就是奔着开源去的，如果这项研究工作能帮助到后来的学者、读者、萌新小白，那这就是这篇文章留存世间的唯一意义，也是我存在的意义！

*曾经搞过三年科研，说实话，在DRL领域，不开源代码的问题一直被人所诟病。我就曾读过假文章，被恶心坏了，作者给的假结果，也不开源代码，为了复现把我几个月时间搭进去了，被白白浪费时间。我是真吐了，我深知有这种恶心人的东西，所以，我这项工作虽说谈不上多厉害，也是坨答辩，但是，问题是你架不住它真啊，它是坨真·答辩。虽然是一坨，但我觉得它是可以帮到你的，绝对保真，只要能帮到你，那它算就是托答辩，也是有价值的！*

视频演示如下：

https://github.com/henbudidiao/UAV-path-planning/assets/64433060/713afd70-6fa0-414e-9939-f106be09c186

## 多无人机行为分析

图5-17以MASAC算法训练的效果为例，展示了多无人机在一个Monte Carlo测试回合中不同时刻的行为 (其中，t<sub>1</sub> < t<sub>2</sub> < t<sub>3</sub> < t<sub>4</sub> < t<sub>5</sub> < t<sub>6</sub>)。
<div align=center>
<img src="https://github.com/user-attachments/assets/7043e8fb-7462-498d-887e-8688692fb59f" style="width:65%;height:65%;">
<img src="https://github.com/user-attachments/assets/6bf892d9-237b-44d5-b7d4-7062cce643cc" style="width:65%;height:65%;">

图5-17 多无人机不同时刻协同航路规划轨迹
</div>

如图5-17所示，领导者UAV、跟随者UAV、目标地点和障碍物在t<sub>1</sub>时刻初始化生成；t<sub>2</sub>时刻领导者UAV有撞上障碍物的碰撞风险，此时领导者UAV需要改变航向进行避障；t<sub>3</sub>时刻领导者UAV避障成功，跟随者UAV逐渐向领导者UAV靠近；t<sub>4</sub>时刻跟随者UAV与领导者UAV成功形成编队；t<sub>5</sub>时刻多无人机持续保持编队，并向目标地点飞行；t<sub>6</sub>时刻多无人机即将到达目标地点。显然，MASAC 算法训练后的多无人机能够在动态不确定的仿真环境中进行协同航路规划，可以在编队保持、安全避障和界内飞行的情形下到达目标地点。

*大伙儿可能会问：“等等，你这敌方无人机哪里来的？你给的开源代码里运行出来没有敌方无人机啊？”*

*答：“这个无足轻重，《信息科学》里就是没有敌方无人机，这里的图来自大论文第五章，此处只是调用大论文里的话来更好的解释多无人机的路径规划情况。”*
