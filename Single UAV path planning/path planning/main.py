# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/7/30 18:13
from rl_env.path_env import RlGame
import numpy as np
from matplotlib import pyplot as plt
import os
import pickle as pkl
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
shoplistfile_test = 'E:\path planning\sample_indextest'  #保存文件数据所在文件的文件名'
N_Agent=1
M_Enemy=5
L_Obstacle=3
RENDER=True
env = RlGame(n=N_Agent,m=M_Enemy,l=L_Obstacle,render=RENDER).unwrapped
EPIOSDE_ALL=500
TEST_EPIOSDE=100
TRAIN_NUM = 5
EP_LEN = 1000
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
LR_A = 5e-4    # learning rate for actor
LR_C = 1e-3    # learning rate for critic
GAMMA = 0.9
MemoryCapacity=20000
Batch=128
Switch=1
tau = 0.0005

def main():
    run(env)
def run(env):
    print('随机测试中...')
    win_times = 0
    average_timestep=0
    average_integral_V=0
    average_integral_U= 0
    all_ep_V, all_ep_U, all_ep_T = [], [], []
    for j in range(TEST_EPIOSDE):
        state = env.reset()
        total_rewards = 0
        integral_V=0
        integral_U=0
        for timestep in range(EP_LEN):
            for i in range(N_Agent+M_Enemy):
                action = env.action_space.sample()
            new_state, reward,done,edge_r,obstacle_r,goal_r,win= env.step(action)  # 执行动作
            if win:
                win_times += 1
            integral_V += state[2]
            integral_U += abs(action).sum()
            total_rewards += reward
            state = new_state
            if RENDER:
                env.render()
            if done:
                break
        average_timestep += timestep
        average_integral_V += integral_V
        average_integral_U += integral_U
        print("Score", total_rewards)
        all_ep_V.append(integral_V)
        all_ep_U.append(integral_U)
        all_ep_T.append(timestep)
    print('任务完成率', win_times / TEST_EPIOSDE)
    print('平均最短飞行时间', average_timestep / TEST_EPIOSDE)
    print('平均最短飞行路程', average_integral_V / TEST_EPIOSDE)
    print('平均最小能量损耗', average_integral_U / TEST_EPIOSDE)
    # env.close()
    # d = {"all_ep_V": all_ep_V, "all_ep_U": all_ep_U, "all_ep_T": all_ep_T}
    # f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
    # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
    # f.close()

    # win_times = 0
    # average_timestep=0
    # average_integral_V=0
    # average_integral_U= 0
    # all_ep_r = [[] for i in range(TRAIN_NUM)]
    # for k in range(TRAIN_NUM):
    #     for j in range(TEST_EPIOSDE):
    #         state = env.reset()
    #         total_rewards = 0
    #         integral_V=0
    #         integral_U=0
    #         for timestep in range(EP_LEN):
    #             for i in range(N_Agent+M_Enemy):
    #                 action = env.action_space.sample()
    #             new_state, reward,done,edge_r,obstacle_r,goal_r,win= env.step(action)  # 执行动作
    #             if win:
    #                 win_times += 1
    #             integral_V += state[2]
    #             integral_U += abs(action).sum()
    #             total_rewards += reward
    #             state = new_state
    #             if RENDER:
    #                 env.render()
    #             if done:
    #                 break
    #         average_timestep += timestep
    #         average_integral_V += integral_V
    #         average_integral_U += integral_U
    #         print("Score", total_rewards)
    #         all_ep_r[k].append(total_rewards)
    #     print('任务完成率', win_times / TEST_EPIOSDE)
    #     print('平均最短飞行时间', average_timestep / TEST_EPIOSDE)
    #     print('平均最短飞行路程', average_integral_V / TEST_EPIOSDE)
    #     print('平均最小能量损耗', average_integral_U / TEST_EPIOSDE)
    #     # env.close()
    # all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
    # all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
    # d = {"all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std}
    # f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
    # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
    # f.close()
    # all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
    # all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
    # plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='随机策略', color='#e75840')
    # plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
    # plt.xlabel('Monte Carlo测试回合数')
    # plt.ylabel('总奖励')
    # plt.legend()
    # plt.show()
if __name__ == '__main__':
    main()