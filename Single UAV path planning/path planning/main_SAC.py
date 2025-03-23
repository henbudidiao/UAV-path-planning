# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/7/30 18:13
from rl_env.path_env import RlGame
# import pygame
# from assignment import constants as C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import os
# import random
import pickle as pkl
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
shoplistfile = 'E:\path planning\SAC'  #保存文件数据所在文件的文件名
shoplistfile_test = 'E:\path planning\SAC_indextest'  #保存文件数据所在文件的文件名'
N_Agent=1
M_Enemy=1
L_Obstacle=1
RENDER=True
env = RlGame(n=N_Agent,m=M_Enemy,l=L_Obstacle,render=RENDER).unwrapped
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
EP_MAX = 500
TRAIN_NUM = 3
TEST_EPIOSDE=100
EP_LEN = 1000
GAMMA = 0.9
q_lr = 3e-4
value_lr = 3e-3
policy_lr = 1e-3
BATCH = 128
tau = 1e-2
MemoryCapacity=20000
Switch=1
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
class ActorNet(nn.Module):
    def __init__(self,inp,outp):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(inp,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,outp)
        self.out.weight.data.normal_(0,0.1)
        self.std_out = nn.Linear(256, outp)
        self.std_out.weight.data.normal_(0, 0.1)

    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=F.relu(inputstate)
        mean=max_action*torch.tanh(self.out(inputstate))#输出概率分布的均值mean
        log_std=self.std_out(inputstate)#softplus激活函数的值域>0
        log_std=torch.clamp(log_std,-20,2)
        std=log_std.exp()
        return mean,std

class CriticNet(nn.Module):
    def __init__(self,input,output):
        super(CriticNet, self).__init__()
        #q1
        self.in_to_y1=nn.Linear(input+output,256)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(256,256)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(256,action_number)
        self.out.weight.data.normal_(0,0.1)
        #q2
        self.q2_in_to_y1 = nn.Linear(input+output, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, action_number)
        self.q2_out.weight.data.normal_(0, 0.1)
    def forward(self,s,a):
        inputstate = torch.cat((s, a), dim=1)
        #q1
        q1=self.in_to_y1(inputstate)
        q1=F.relu(q1)
        q1=self.y1_to_y2(q1)
        q1=F.relu(q1)
        q1=self.out(q1)
        #q2
        q2 = self.q2_in_to_y1(inputstate)
        q2 = F.relu(q2)
        q2 = self.q2_y1_to_y2(q2)
        q2 = F.relu(q2)
        q2 = self.q2_out(q2)
        return q1,q2

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
class Actor():
    def __init__(self):
        self.action_net=ActorNet(state_number,action_number).cuda()#这只是均值mean
        self.optimizer=torch.optim.Adam(self.action_net.parameters(),lr=policy_lr)

    def choose_action(self,s):
        inputstate = torch.FloatTensor(s).cuda()
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()
        action=torch.clamp(action,min_action,max_action)
        return action.cpu().detach().numpy()
    def evaluate(self,s):
        # inputstate = torch.FloatTensor(s).cuda()
        mean,std=self.action_net(s)
        dist = torch.distributions.Normal(mean, std)
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample()
        action=torch.tanh(mean+std*z)
        action=torch.clamp(action,min_action,max_action)
        action_logprob=dist.log_prob(mean+std*z)-torch.log(1-action.pow(2)+1e-6)
        return action,action_logprob

    def learn(self,actor_loss):
        loss=actor_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Entroy():
    def __init__(self):
        self.target_entropy = -0.1
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=q_lr)

    def learn(self,entroy_loss):
        loss=entroy_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Critic():
    def __init__(self):
        self.critic_v,self.target_critic_v=CriticNet(state_number,action_number).cuda(),CriticNet(state_number,action_number).cuda()#改网络输入状态，生成一个Q值
        self.target_critic_v.load_state_dict(self.critic_v.state_dict())
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=value_lr,eps=1e-5)
        self.lossfunc = nn.MSELoss()
    def soft_update(self):
        for target_param, param in zip(self.target_critic_v.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_v(self,s,a):
        return self.critic_v(s,a)

    def target_get_v(self,s,a):
        return self.target_critic_v(s,a)

    def learn(self,current_q1,current_q2,target_q):
        loss = self.lossfunc(current_q1, target_q) + self.lossfunc(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def main():
    run(env)
def run(env):
    if Switch==0:
        print('SAC训练中...')
        all_ep_r = [[] for i in range(TRAIN_NUM)]
        all_ep_r_edge = [[] for i in range(TRAIN_NUM)]
        all_ep_r_obstacle = [[] for i in range(TRAIN_NUM)]
        all_ep_r_goal = [[] for i in range(TRAIN_NUM)]
        for k in range(TRAIN_NUM):
            actor = Actor()
            critic = Critic()
            entroy=Entroy()
            M = Memory(MemoryCapacity, 2 * state_number + action_number + 1)
            ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(action_number))
            # all_ep_r = []
            for episode in range(EP_MAX):
                observation = env.reset()  # 环境重置
                reward_totle, reward_totle_edge, reward_totle_obstacle, reward_totle_goal = 0,0,0,0
                for timestep in range(EP_LEN):
                    action = actor.choose_action(observation)
                    if episode <= 50:
                        noise = ou_noise()
                    else:
                        noise = 0
                    action = action + noise
                    action = np.clip(action, -max_action, max_action)
                    observation_, reward,done,edge_r,obstacle_r,goal_r,win= env.step(action)  # 单步交互
                    M.store_transition(observation, action, reward, observation_)
                    # 记忆库存储
                    # 有的2000个存储数据就开始学习
                    if M.memory_counter > MemoryCapacity:
                        b_M = M.sample(BATCH)
                        b_s = b_M[:, :state_number]
                        b_a = b_M[:, state_number: state_number + action_number]
                        b_r = b_M[:, -state_number - 1: -state_number]
                        b_s_ = b_M[:, -state_number:]
                        b_s = torch.FloatTensor(b_s).cuda()
                        b_a = torch.FloatTensor(b_a).cuda()
                        b_r = torch.FloatTensor(b_r).cuda()
                        b_s_ = torch.FloatTensor(b_s_).cuda()
                        new_action, log_prob_ = actor.evaluate(b_s_)
                        target_q1,target_q2=critic.target_critic_v(b_s_,new_action)
                        target_q=b_r+GAMMA*(torch.min(target_q1,target_q2)-entroy.alpha.cuda()*log_prob_)
                        current_q1, current_q2 = critic.get_v(b_s, b_a)
                        critic.learn(current_q1,current_q2,target_q.detach())
                        a,log_prob=actor.evaluate(b_s)
                        q1,q2=critic.get_v(b_s,a)
                        q=torch.min(q1,q2)
                        actor_loss = (entroy.alpha.cuda() * log_prob - q).mean()
                        actor.learn(actor_loss)
                        alpha_loss = -(entroy.log_alpha.exp().cuda() * (log_prob + entroy.target_entropy).detach()).mean()
                        entroy.learn(alpha_loss)
                        entroy.alpha=entroy.log_alpha.exp().cuda()
                        # 软更新
                        critic.soft_update()
                    observation = observation_
                    reward_totle += reward
                    reward_totle_edge += edge_r
                    reward_totle_obstacle += obstacle_r
                    reward_totle_goal += (goal_r+obstacle_r)
                    if RENDER:
                        env.render()
                    if done:
                        break
                print("Ep: {} rewards: {}".format(episode, reward_totle))
                # all_ep_r.append(reward_totle)
                all_ep_r[k].append(reward_totle)
                all_ep_r_edge[k].append(reward_totle_edge)
                all_ep_r_obstacle[k].append(reward_totle_obstacle)
                all_ep_r_goal[k].append(reward_totle_goal)
                if episode % 20 == 0 and episode > 200:#保存神经网络参数
                    save_data = {'net': actor.action_net.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': episode}
                    torch.save(save_data, "E:\path planning\Path_SAC_actor.pth")
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
        print('SAC测试中...')
        aa = Actor()
        checkpoint_aa = torch.load("E:\path planning\Path_SAC_actor.pth")
        aa.action_net.load_state_dict(checkpoint_aa['net'])
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
            for timestep in range(EP_LEN):
                action = aa.choose_action(state)
                new_state, reward,done ,edge_r,obstacle_r,goal_r,win= env.step(action)  # 执行动作
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
        # d = {"all_ep_V": all_ep_V, "all_ep_U": all_ep_U, "all_ep_T": all_ep_T}
        # f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
        # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
        # f.close()
        plt.show()
if __name__ == '__main__':
    main()