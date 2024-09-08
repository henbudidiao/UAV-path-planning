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
import pickle as pkl
shoplistfile = 'G:\path planning\MADDPG'  #保存文件数据所在文件的文件名
shoplistfile_test = 'G:\path planning\MADDPG_compare'  #保存文件数据所在文件的文件名
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Agent=1
M_Enemy=4
RENDER=True
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped
state_number=7
TEST_EPIOSDE=100
TRAIN_NUM = 3
EP_LEN = 1000
EPIOSDE_ALL=500
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
LR_A = 1e-3    # learning rate for actor
LR_C = 1e-3    # learning rate for critic
GAMMA = 0.95
MemoryCapacity=20000
Batch=128
Switch=1
tau = 0.005
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
        self.in_to_y1=nn.Linear(input+output,40)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(40,20)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(20,1)
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
        self.critic_estimate_eval,self.critic_reality_target=CriticNet(state_number*(N_Agent+M_Enemy),action_number),CriticNet(state_number*(N_Agent+M_Enemy),action_number)
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
        all_ep_r0 = [[] for i in range(TRAIN_NUM)]
        all_ep_r1 = [[] for i in range(TRAIN_NUM)]
        for k in range(TRAIN_NUM):
            actors = [None for _ in range(N_Agent + M_Enemy)]
            critics = [None for _ in range(N_Agent + M_Enemy)]
            for i in range(N_Agent + M_Enemy):
                actors[i] = Actor()
                critics[i] = Critic()
            M = Memory(MemoryCapacity, 2 * state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy) + 1*(N_Agent+M_Enemy))  # 奖惩是一个浮点数
            ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(((N_Agent+M_Enemy), action_number)))
            action = np.zeros(((N_Agent + M_Enemy), action_number))
            # all_ep_r = []
            for episode in range(EPIOSDE_ALL):
                observation=env.reset()
                reward_totle, reward_totle0, reward_totle1 = 0, 0, 0
                for timestep in range(EP_LEN):
                    for i in range(N_Agent + M_Enemy):
                        action[i] = actors[i].choose_action(observation[i])
                        # action[0]=actor0.choose_action(observation[0])
                        # action[1] = actor0.choose_action(observation[1])
                    if episode <= 50:
                        noise = ou_noise()
                    else:
                        noise = 0
                    action = action + noise
                    action = np.clip(action, -max_action, max_action)
                    observation_, reward,done,win,team_counter = env.step(action)  # 单步交互
                    M.store_transition(observation.flatten(), action.flatten(), reward.flatten()/1000, observation_.flatten())
                    if M.memory_counter > MemoryCapacity:
                        b_M = M.sample(Batch)
                        b_s = b_M[:, :state_number*(N_Agent+M_Enemy)]
                        b_a = b_M[:,
                              state_number * (N_Agent + M_Enemy): state_number * (N_Agent + M_Enemy) + action_number * (
                                          N_Agent + M_Enemy)]
                        b_r = b_M[:, -state_number * (N_Agent + M_Enemy) - 1 * (N_Agent + M_Enemy): -state_number * (
                                    N_Agent + M_Enemy)]
                        b_s_ = b_M[:, -state_number * (N_Agent + M_Enemy):]
                        for i in range(N_Agent + M_Enemy):
                            actor_action_0 = actors[i].learn_a(b_s[:, state_number*i:state_number*(i+1)])
                            # actor_action_1 = actors[1].learn_a(b_s[:, state_number:state_number * 2])
                            # actor_action = torch.hstack((actor_action_0, actor_action_1))
                            actor_action_0_ = actors[i].learn_a_(b_s_[:, state_number*i:state_number*(i+1)])
                            # actor_action_1_ = actors[1].learn_a_(b_s_[:, state_number:state_number * 2])
                            # actor_action_ = torch.hstack((actor_action_0_, actor_action_1_))
                            critics[i].learn(b_s, b_a[:, action_number*i:action_number*(i+1)], b_r, b_s_, actor_action_0_)
                            Q_c_to_a_loss = critics[i].learn_loss(b_s, actor_action_0)
                            actors[i].learn(Q_c_to_a_loss)
                            # 软更新
                            actors[i].soft_update()
                            critics[i].soft_update()
                    observation = observation_
                    reward_totle += reward.mean()
                    reward_totle0 += float(reward[0])
                    reward_totle1 += float(reward[1])
                    if RENDER:
                        env.render()
                    if done:
                        break
                print('Episode {}，奖励：{}'.format(episode, reward_totle))
                # all_ep_r.append(reward_totle)
                all_ep_r[k].append(reward_totle)
                all_ep_r0[k].append(reward_totle0)
                all_ep_r1[k].append(reward_totle1)
                if episode % 50 == 0 and episode > 200:#保存神经网络参数
                    save_data = {'net': actors[0].actor_estimate_eval.state_dict(), 'opt': actors[0].optimizer.state_dict()}
                    torch.save(save_data, "G:\path planning\Path_DDPG_actor_new.pth")
                    save_data = {'net': actors[1].actor_estimate_eval.state_dict(), 'opt': actors[1].optimizer.state_dict()}
                    torch.save(save_data, "G:\path planning\Path_DDPG_actor_1_new.pth")
            # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
            # plt.xlabel('Episode')
            # plt.ylabel('Moving averaged episode reward')
            # plt.show()
            # env.close()
        all_ep_r_mean = np.mean((np.array(all_ep_r)), axis=0)
        all_ep_r_std = np.std((np.array(all_ep_r)), axis=0)
        all_ep_L_mean = np.mean((np.array(all_ep_r0)), axis=0)
        all_ep_L_std = np.std((np.array(all_ep_r0)), axis=0)
        all_ep_F_mean = np.mean((np.array(all_ep_r1)), axis=0)
        all_ep_F_std = np.std((np.array(all_ep_r1)), axis=0)
        d = {"all_ep_r_mean": all_ep_r_mean, "all_ep_r_std": all_ep_r_std,
             "all_ep_L_mean": all_ep_L_mean, "all_ep_L_std": all_ep_L_std,
             "all_ep_F_mean": all_ep_F_mean, "all_ep_F_std": all_ep_F_std,}
        f = open(shoplistfile, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
        pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
        f.close()
        all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
        all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
        all_ep_L_max = all_ep_L_mean + all_ep_L_std * 0.95
        all_ep_L_min = all_ep_L_mean - all_ep_L_std * 0.95
        all_ep_F_max = all_ep_F_mean + all_ep_F_std * 0.95
        all_ep_F_min = all_ep_F_mean - all_ep_F_std * 0.95
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='MADDPG', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Total reward')
        plt.figure(2, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, label='MADDPG', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Leader reward')
        plt.figure(3, figsize=(8, 4), dpi=150)
        plt.margins(x=0)
        plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, label='MADDPG', color='#e75840')
        plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, alpha=0.6,
                         facecolor='#e75840')
        plt.xlabel('Episode')
        plt.ylabel('Follower reward')
        plt.legend()
        plt.show()
        env.close()
    else:
        print('MADDPG测试中...')
        aa = Actor()
        checkpoint_aa = torch.load("G:\path planning\Path_DDPG_actor_new.pth")
        aa.actor_estimate_eval.load_state_dict(checkpoint_aa['net'])
        bb = Actor()
        checkpoint_bb = torch.load("G:\path planning\Path_DDPG_actor_1_new.pth")
        bb.actor_estimate_eval.load_state_dict(checkpoint_bb['net'])
        action = np.zeros((N_Agent + M_Enemy, action_number))
        win_times = 0
        average_FKR = 0
        average_timestep = 0
        average_integral_V = 0
        average_integral_U = 0
        all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []
        for j in range(TEST_EPIOSDE):
            state = env.reset()
            total_rewards = 0
            integral_V = 0
            integral_U = 0
            v, v1 = [], []
            for timestep in range(EP_LEN):
                for i in range(N_Agent):
                    action[i] = aa.choose_action(state[i])
                for i in range(M_Enemy):
                    action[i + 1] = bb.choose_action(state[i + 1])
                # action[0] = aa.choose_action(state[0])
                # action[1] = bb.choose_action(state[1])
                new_state, reward, done, win, team_counter,d = env.step(action)  # 执行动作
                if win:
                    win_times += 1
                v.append(state[0][2])
                v1.append(state[1][2])
                integral_V += state[0][2]
                integral_U += abs(action[0]).sum()
                total_rewards += reward.mean()
                state = new_state
                if RENDER:
                    env.render()
                if done:
                    break
            FKR = team_counter / timestep
            average_FKR += FKR
            average_timestep += timestep
            average_integral_V += integral_V
            average_integral_U += integral_U
            print("Score", total_rewards)
            all_ep_V.append(integral_V)
            all_ep_U.append(integral_U)
            all_ep_T.append(timestep)
            all_ep_F.append(FKR)
            # print('最大编队保持率',FKR)
            # print('最短飞行时间',timestep)
            # print('最短飞行路程', integral_V)
            # print('最小能量损耗', integral_U)
            # plt.plot(np.arange(len(v)), v)
            # plt.plot(np.arange(len(v1)), v1)
            # plt.show()
        print('任务完成率', win_times / TEST_EPIOSDE)
        print('平均最大编队保持率', average_FKR / TEST_EPIOSDE)
        print('平均最短飞行时间', average_timestep / TEST_EPIOSDE)
        print('平均最短飞行路程', average_integral_V / TEST_EPIOSDE)
        print('平均最小能量损耗', average_integral_U / TEST_EPIOSDE)
        d = {"all_ep_V": all_ep_V, "all_ep_U": all_ep_U, "all_ep_T": all_ep_T, "all_ep_F": all_ep_F, }
        f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
        pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
        f.close()
        env.close()
if __name__ == '__main__':
    main()