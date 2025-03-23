# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/7/20 23:34
from rl_env.path_env import RlGame
# import pygame
# from assignment import constants as C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import gym
import pickle as pkl
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
shoplistfile = 'E:\path planning\DDPG1'  #保存文件数据所在文件的文件名
shoplistfile_test = 'E:\path planning\DDPG_indextest'  #保存文件数据所在文件的文件名'
N_Agent=1
M_Enemy=1
L_Obstacle=1
RENDER=False
env = RlGame(n=N_Agent,m=M_Enemy,l=L_Obstacle,render=RENDER).unwrapped
EPIOSDE_ALL=500
TEST_EPIOSDE=100
TRAIN_NUM = 1
EP_LEN = 1000
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
LR_A = 5e-4    # learning rate for actor
LR_C = 1e-3    # learning rate for critic
GAMMA = 0.9
# reward discount
MemoryCapacity=20000
Batch=128
Switch=0
tau = 0.0005
'''DDPG第一步 设计A-C框架的Actor（DDPG算法，只有critic的部分才会用到记忆库）'''
'''第一步 设计A-C框架形式的网络部分'''
class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,50)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(50,20)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(20,outp)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=torch.sigmoid(inputstate)
        act=max_action*torch.tanh(self.out(inputstate))
        # return F.softmax(act,dim=-1)
        return act
class CriticNet(nn.Module):
    def __init__(self,input,output):
        super(CriticNet, self).__init__()
        self.in_to_y1=nn.Linear(input,40)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(40,20)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(20,output)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,s,a):
        inputstate = torch.cat((s, a), dim=1)
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=torch.sigmoid(inputstate)
        Q=self.out(inputstate)
        return Q
class Actor():
    def __init__(self):
        self.actor_estimate_eval,self.actor_reality_target = ActorNet(state_number,action_number),ActorNet(state_number,action_number)
        self.optimizer = torch.optim.Adam(self.actor_estimate_eval.parameters(), lr=LR_A)

    '''第二步 编写根据状态选择动作的函数'''

    def choose_action(self, s):
        inputstate = torch.FloatTensor(s)
        probs = self.actor_estimate_eval(inputstate)
        return probs.detach().numpy()

    '''第四步 编写A的学习函数'''
    '''生成输入为s的actor估计网络，用于传给critic估计网络，虽然这与choose_action函数一样，但如果直接用choose_action
    函数生成的动作，DDPG是不会收敛的，原因在于choose_action函数生成的动作经过了记忆库，动作从记忆库出来后，动作的梯度数据消失了
    所以再次编写了learn_a函数，它生成的动作没有过记忆库，是带有梯度的'''

    def learn_a(self, s):
        s = torch.FloatTensor(s)
        A_prob = self.actor_estimate_eval(s)
        return A_prob

    '''把s_输入给actor现实网络，生成a_，a_将会被传给critic的实现网络'''

    def learn_a_(self, s_):
        s_ = torch.FloatTensor(s_)
        A_prob=self.actor_reality_target(s_).detach()
        return A_prob

    '''actor的学习函数接受来自critic估计网络算出的Q_estimate_eval当做自己的loss，即负的critic_estimate_eval(s,a)，使loss
    最小化，即最大化critic网络生成的价值'''

    def learn(self, a_loss):
        loss = a_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''第六步，最后一步  编写软更新程序，Actor部分与critic部分都会有软更新代码'''
    '''DQN是硬更新，即固定时间更新，而DDPG采用软更新，w_老_现实=τ*w_新_估计+(1-τ)w_老_现实'''
    def soft_update(self):
        for target_param, param in zip(self.actor_reality_target.parameters(), self.actor_estimate_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Critic():
    def __init__(self):
        self.critic_estimate_eval,self.critic_reality_target=CriticNet(state_number+action_number,1),CriticNet(state_number+action_number,1)
        self.optimizer = torch.optim.Adam(self.critic_estimate_eval.parameters(), lr=LR_C)
        self.lossfun=nn.MSELoss()

    '''第五步 编写critic的学习函数'''
    '''使用critic估计网络得到 actor的loss，这里的输入参数a是带梯度的'''

    def learn_loss(self, s, a):
        s = torch.FloatTensor(s)
        # a = a.view(-1, 1)
        Q_estimate_eval = -self.critic_estimate_eval(s, a).mean()
        return Q_estimate_eval

    '''这里的输入参数a与a_是来自记忆库的，不带梯度，根据公式我们会得到critic的loss'''

    def learn(self, s, a, r, s_, a_):
        s = torch.FloatTensor(s)
        a = torch.FloatTensor(a)#当前动作a来自记忆库
        r = torch.FloatTensor(r)
        s_ = torch.FloatTensor(s_)
        # a_ = a_.view(-1, 1)  # view中一个参数定为-1，代表动态调整这个维度上的元素个数，以保证元素的总数不变
        Q_estimate_eval = self.critic_estimate_eval(s, a)
        Q_next = self.critic_reality_target(s_, a_).detach()
        Q_reality_target = r + GAMMA * Q_next
        loss = self.lossfun(Q_estimate_eval, Q_reality_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self):
        for target_param, param in zip(self.critic_reality_target.parameters(), self.critic_estimate_eval.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

'''第三步  建立记忆库'''
class Memory():
    def __init__(self,capacity,dims):
        self.capacity=capacity
        self.mem=np.zeros((capacity,dims))
        self.memory_counter=0
    '''存储记忆'''
    def store_transition(self,s,a,r,s_):
        tran = np.hstack((s, a,r, s_))  # 把s,a,r,s_困在一起，水平拼接
        index = self.memory_counter % self.capacity#除余得索引
        self.mem[index, :] = tran  # 给索引存值，第index行所有列都为其中一次的s,a,r,s_；mem会是一个capacity行，（s+a+r+s_）列的数组
        self.memory_counter+=1
    '''随机从记忆库里抽取'''
    def sample(self,n):
        assert self.memory_counter>=self.capacity,'记忆库没有存满记忆'
        sample_index = np.random.choice(self.capacity, n)#从capacity个记忆里随机抽取n个为一批，可得到抽样后的索引号
        new_mem = self.mem[sample_index, :]#由抽样得到的索引号在所有的capacity个记忆中  得到记忆s，a，r，s_
        return new_mem
    '''OU噪声'''

class Ornstein_Uhlenbeck_Noise:
    def __init__(self, mu, sigma=0.1, theta=0.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + \
            self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        '''
        后两行是dXt，其中后两行的前一行是θ(μ-Xt)dt，后一行是σεsqrt(dt)
        '''
        self.x_prev = x
        return x

    def reset(self):
        if self.x0 is not None:
            self.x_prev = self.x0
        else:
            self.x_prev = np.zeros_like(self.mu)

def main():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    run(env)

def run(env):
    if Switch==0:
        all_ep_r = [[] for i in range(TRAIN_NUM)]
        all_ep_r_edge = [[] for i in range(TRAIN_NUM)]
        all_ep_r_obstacle = [[] for i in range(TRAIN_NUM)]
        all_ep_r_goal = [[] for i in range(TRAIN_NUM)]
        for k in range(TRAIN_NUM):
            actor = Actor()
            critic = Critic()
            M = Memory(MemoryCapacity, 2 * state_number + action_number + 1)  # 奖惩是一个浮点数
            ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(action_number))
            # all_ep_r = []
            for episode in range(EPIOSDE_ALL):
                obs=env.reset()
                reward_totle, reward_totle_edge, reward_totle_obstacle, reward_totle_goal = 0, 0, 0, 0
                for timestep in range(EP_LEN):
                    action = actor.choose_action(obs)
                    if episode<=50:
                        noise=np.random.normal(loc=0.0, scale=1)
                        # noise=ou_noise()
                    else:
                        noise=0
                    action=action+noise
                    action=np.clip(action,-max_action,max_action)
                    # all_action.append(action)
                    # action=env.action_space.sample()
                    # if not first_flag:
                    obs_,reward,done,edge_r,obstacle_r,goal_r,win= env.step(action)
                    M.store_transition(obs, action, reward, obs_)
                    if M.memory_counter > MemoryCapacity:
                        b_M = M.sample(Batch)
                        b_s = b_M[:, :state_number]
                        b_a = b_M[:, state_number: state_number + action_number]
                        b_r = b_M[:, -state_number - 1: -state_number]
                        b_s_ = b_M[:, -state_number:]
                        actor_action = actor.learn_a(b_s)
                        actor_action_ = actor.learn_a_(b_s_)
                        critic.learn(b_s, b_a, b_r, b_s_, actor_action_)
                        Q_c_to_a_loss = critic.learn_loss(b_s, actor_action)
                        actor.learn(Q_c_to_a_loss)
                        # 软更新
                        actor.soft_update()
                        critic.soft_update()
                    obs = obs_
                    reward_totle += reward
                    reward_totle_edge += edge_r
                    reward_totle_obstacle += obstacle_r
                    reward_totle_goal +=  (goal_r+obstacle_r)
                    if RENDER:
                        env.render()
                    if done:
                        break
                print('Episode {}，奖励：{}'.format(episode, reward_totle))
                # all_ep_r.append(reward_totle)
                # all_ep_r[k].append(reward_totle)
                all_ep_r_edge[k].append(reward_totle_edge)
                all_ep_r_obstacle[k].append(reward_totle_obstacle)
                all_ep_r_goal[k].append(reward_totle_goal)
                # plt.plot(np.arange(len(all_action)), all_action)
                # plt.show()
                if episode == 0:
                    all_ep_r[k].append(reward_totle)
                else:
                    all_ep_r[k].append(all_ep_r[k][-1] * 0.9 + reward_totle * 0.1)
                if episode % 50 == 0 and episode > 200:#保存神经网络参数
                    save_data = {'net': actor.actor_estimate_eval.state_dict(), 'opt': actor.optimizer.state_dict()}
                    torch.save(save_data, "E:\path planning\Path_DDPG_actor1.pth")
            # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            # plt.xlabel('Episode')
            # plt.ylabel('Moving averaged episode reward')
            # plt.show()
            # env.close()
        all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
        all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
        all_ep_edge_mean = np.mean((np.array(all_ep_r_edge)), axis=0)
        all_ep_edge_std = np.std((np.array(all_ep_r_edge)), axis=0)
        all_ep_obstacle_mean = np.mean((np.array(all_ep_r_obstacle)), axis=0)
        all_ep_obstacle_std = np.std((np.array(all_ep_r_obstacle)), axis=0)
        all_ep_goal_mean = np.mean((np.array(all_ep_r_goal)), axis=0)
        all_ep_goal_std = np.std((np.array(all_ep_r_goal)), axis=0)
        d = {"all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std,
             "all_ep_edge_mean": all_ep_edge_mean, "all_ep_edge_std": all_ep_edge_std,
             "all_ep_obstacle_mean": all_ep_obstacle_mean, "all_ep_obstacle_std": all_ep_obstacle_std,
             "all_ep_goal_mean": all_ep_goal_mean, "all_ep_goal_std": all_ep_goal_std,
             }
        f = open(shoplistfile, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
        pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
        f.close()
        all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
        all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
        all_ep_edge_max = all_ep_edge_mean + all_ep_edge_std * 0.95
        all_ep_edge_min = all_ep_edge_mean - all_ep_edge_std * 0.95
        all_ep_obstacle_max = all_ep_obstacle_mean + all_ep_obstacle_std * 0.95
        all_ep_obstacle_min = all_ep_obstacle_mean - all_ep_obstacle_std * 0.95
        all_ep_goal_max = all_ep_goal_mean + all_ep_goal_std * 0.95
        all_ep_goal_min = all_ep_goal_mean - all_ep_goal_std * 0.95
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='SAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.figure(2, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_edge_mean)), all_ep_edge_mean, label='SAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_edge_mean)), all_ep_edge_max, all_ep_edge_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Edge reward')
        plt.figure(3, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_obstacle_mean)), all_ep_obstacle_mean, label='SAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_obstacle_mean)), all_ep_obstacle_max, all_ep_obstacle_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Obstacle reward')
        plt.figure(4, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_goal_mean)), all_ep_goal_mean, label='SAC', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_goal_mean)), all_ep_goal_max, all_ep_goal_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Goal reward')
        # plt.legend()
        plt.show()
        env.close()
    else:
        print('DDPG测试中...')
        aa=Actor()
        checkpoint_aa = torch.load("E:\path planning\Path_DDPG_actor.pth")
        aa.actor_estimate_eval.load_state_dict(checkpoint_aa['net'])
        win_times = 0
        average_timestep=0
        average_integral_V=0
        average_integral_U= 0
        all_ep_V, all_ep_U, all_ep_T = [], [], []
        for j in range(TEST_EPIOSDE):
            state = env.reset()
            total_rewards = 0
            integral_V = 0
            integral_U = 0
            v, v1 = [], []
            for timestep in range(EP_LEN):
                action=aa.choose_action(state)
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
        print('任务完成率',win_times / TEST_EPIOSDE)
        print('平均最短飞行时间', average_timestep/TEST_EPIOSDE)
        print('平均最短飞行路程', average_integral_V/TEST_EPIOSDE)
        print('平均最小能量损耗', average_integral_U/TEST_EPIOSDE)
        # env.close()
        d = {"all_ep_V": all_ep_V, "all_ep_U": all_ep_U, "all_ep_T": all_ep_T}
        # f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
        # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
        # f.close()
        plt.show()
if __name__ == '__main__':
    main()