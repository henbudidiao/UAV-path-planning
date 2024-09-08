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
import os
import pickle as pkl
shoplistfile = 'G:\path planning\MASAC_new1'  #保存文件数据所在文件的文件名
shoplistfile_test = 'G:\path planning\MASAC_d_test2'  #保存文件数据所在文件的文件名
shoplistfile_test1 = 'G:\path planning\MASAC_compare'  #保存文件数据所在文件的文件名
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
N_Agent=1
M_Enemy=4
RENDER=True
TRAIN_NUM = 1
TEST_EPIOSDE=100
env = RlGame(n=N_Agent,m=M_Enemy,render=RENDER).unwrapped
state_number=7
action_number=env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
EP_MAX = 500
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
        self.out=nn.Linear(256,1)
        self.out.weight.data.normal_(0,0.1)
        #q2
        self.q2_in_to_y1 = nn.Linear(input+output, 256)
        self.q2_in_to_y1.weight.data.normal_(0, 0.1)
        self.q2_y1_to_y2 = nn.Linear(256, 256)
        self.q2_y1_to_y2.weight.data.normal_(0, 0.1)
        self.q2_out = nn.Linear(256, 1)
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
        self.action_net=ActorNet(state_number,action_number)#这只是均值mean
        self.optimizer=torch.optim.Adam(self.action_net.parameters(),lr=policy_lr)

    def choose_action(self,s):
        inputstate = torch.FloatTensor(s)
        mean,std=self.action_net(inputstate)
        dist = torch.distributions.Normal(mean, std)
        action=dist.sample()
        action=torch.clamp(action,min_action,max_action)
        return action.detach().numpy()
    def evaluate(self,s):
        inputstate = torch.FloatTensor(s)
        mean,std=self.action_net(inputstate)
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
        self.critic_v,self.target_critic_v=CriticNet(state_number*(N_Agent+M_Enemy),action_number),CriticNet(state_number*(N_Agent+M_Enemy),action_number)#改网络输入状态，生成一个Q值
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
        try:
            assert M_Enemy == 1
        except:
            print('程序终止，被逮到~嘿嘿，哥们儿预判到你会犯错，这段程序中变量\'M_Enemy\'的值必须为1，请把它的值改为1。\n' 
                  '改为1之后程序一定会报错，这是因为组数越界，更改path_env.py文件中的跟随者无人机初始化个数；删除多余的\n'
                  '求距离函数，即变量dis_1_agent_0_to_3等，以及提到变量dis_1_agent_0_to_3等的地方；删除画无人机轨迹的\n'
                  '函数；删除step函数的最后一个返回值dis_1_agent_0_to_1；将player.py文件中的变量dt改为1；即可开始训练！\n'
                  '如果实在不会改也无妨，我会在不久之后出一个视频来手把手教大伙怎么改，可持续关注此项目github中的README文件。\n')
        else:
            print('SAC训练中...')
            all_ep_r = [[] for i in range(TRAIN_NUM)]
            all_ep_r0 = [[] for i in range(TRAIN_NUM)]
            all_ep_r1 = [[] for i in range(TRAIN_NUM)]
            for k in range(TRAIN_NUM):
                actors = [None for _ in range(N_Agent+M_Enemy)]
                critics = [None for _ in range(N_Agent+M_Enemy)]
                entroys = [None for _ in range(N_Agent+M_Enemy)]
                for i in range(N_Agent+M_Enemy):
                    actors[i] = Actor()
                    critics[i] = Critic()
                    entroys[i] = Entroy()
                M = Memory(MemoryCapacity, 2 * state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy) + 1*(N_Agent+M_Enemy))
                ou_noise = Ornstein_Uhlenbeck_Noise(mu=np.zeros(((N_Agent+M_Enemy), action_number)))
                action=np.zeros(((N_Agent+M_Enemy), action_number))
                # aaa = np.zeros((N_Agent, state_number))
                for episode in range(EP_MAX):
                    observation = env.reset()  # 环境重置
                    reward_totle,reward_totle0,reward_totle1 = 0,0,0
                    for timestep in range(EP_LEN):
                        for i in range(N_Agent+M_Enemy):
                            action[i] = actors[i].choose_action(observation[i])
                        # action[0]=actor0.choose_action(observation[0])
                        # action[1] = actor0.choose_action(observation[1])
                        if episode <= 20:
                            noise = ou_noise()
                        else:
                            noise = 0
                        action = action + noise
                        action = np.clip(action, -max_action, max_action)
                        observation_, reward,done,win,team_counter= env.step(action)  # 单步交互
                        M.store_transition(observation.flatten(), action.flatten(), reward.flatten(), observation_.flatten())
                        # 记忆库存储
                        # 有的2000个存储数据就开始学习
                        if M.memory_counter > MemoryCapacity:
                            b_M = M.sample(BATCH)
                            b_s = b_M[:, :state_number*(N_Agent+M_Enemy)]
                            b_a = b_M[:, state_number*(N_Agent+M_Enemy): state_number*(N_Agent+M_Enemy) + action_number*(N_Agent+M_Enemy)]
                            b_r = b_M[:, -state_number*(N_Agent+M_Enemy) - 1*(N_Agent+M_Enemy): -state_number*(N_Agent+M_Enemy)]
                            b_s_ = b_M[:, -state_number*(N_Agent+M_Enemy):]
                            b_s = torch.FloatTensor(b_s)
                            b_a = torch.FloatTensor(b_a)
                            b_r = torch.FloatTensor(b_r)
                            b_s_ = torch.FloatTensor(b_s_)
                            # if not done[0]:
                            #     new_action_0, log_prob_0 = actor0.evaluate(b_s_[:, 0:state_number])
                            #     target_q10, target_q20 = critic0.target_critic_v(b_s_[:, 0:state_number], new_action_0)
                            #     target_q0 = b_r[:, 0:1] + GAMMA * (1 - b_done[0]) *(torch.min(target_q10, target_q20) - entroy0.alpha * log_prob_0)
                            #     current_q10, current_q20 = critic0.get_v(b_s[:,0:state_number], b_a[:, 0:action_number*1])
                            #     critic0.learn(current_q10, current_q20, target_q0.detach())
                            #     a0, log_prob0 = actor0.evaluate(b_s[:, 0:state_number*1])
                            #     q10, q20 = critic0.get_v(b_s[:, 0:state_number*1], a0)
                            #     q0 = torch.min(q10, q20)
                            #     actor_loss0 = (entroy0.alpha * log_prob0 - q0).mean()
                            #     alpha_loss0 = -(entroy0.log_alpha.exp() * (
                            #                     log_prob0 + entroy0.target_entropy).detach()).mean()
                            #     actor0.learn(actor_loss0)
                            #     entroy0.learn(alpha_loss0)
                            #     entroy0.alpha = entroy0.log_alpha.exp()
                            #     # 软更新
                            #     critic0.soft_update()
                            # if not done[1]:
                            #     new_action_1, log_prob_1 = actor1.evaluate(b_s_[:, state_number:state_number*2])
                            #     target_q11, target_q21 = critic1.target_critic_v(b_s_[:, state_number:state_number*2], new_action_1)
                            #     target_q1= b_r[:, 1:2] + GAMMA * (1 - b_done[1])*(torch.min(target_q11, target_q21) - entroy1.alpha * log_prob_1)
                            #     current_q11, current_q21 = critic1.get_v(b_s[:, state_number:state_number*2], b_a[:, action_number:action_number * 2])
                            #     critic1.learn(current_q11, current_q21, target_q1.detach())
                            #     a1, log_prob1 = actor1.evaluate(b_s[:, state_number:state_number*2])
                            #     q11, q21 = critic1.get_v(b_s[:, state_number:state_number*2], a1)
                            #     q1 = torch.min(q11, q21)
                            #     actor_loss1 = (entroy1.alpha * log_prob1 - q1).mean()
                            #     alpha_loss1 = -(entroy1.log_alpha.exp() * (
                            #             log_prob1 + entroy1.target_entropy).detach()).mean()
                            #     actor1.learn(actor_loss1)
                            #     entroy1.learn(alpha_loss1)
                            #     entroy1.alpha = entroy1.log_alpha.exp()
                            #     # 软更新
                            #     critic1.soft_update()
                            for i in range(N_Agent+M_Enemy):
                            # # # TODO 方法二
                            # new_action_0, log_prob_0 = actor0.evaluate(b_s_[:, :state_number])
                            # new_action_1, log_prob_1 = actor0.evaluate(b_s_[:, state_number:state_number * 2])
                            # new_action = torch.hstack((new_action_0, new_action_1))
                            # # new_action = torch.cat((new_action_0, new_action_1),dim=1)
                            # log_prob_ = (log_prob_0 + log_prob_1) / 2
                            # # log_prob_=torch.hstack((log_prob_0.mean(axis=1).unsqueeze(dim=1),log_prob_1.mean(axis=1).unsqueeze(dim=1)))
                            # target_q1, target_q2 = critic0.target_critic_v(b_s_, new_action)
                            #
                            # target_q = b_r + GAMMA * (torch.min(target_q1, target_q2) - entroy0.alpha * log_prob_)
                            #
                            # current_q1, current_q2 = critic0.get_v(b_s, b_a)
                            # critic0.learn(current_q1, current_q2, target_q.detach())
                            # a0, log_prob0 = actor0.evaluate(b_s[:, :state_number])
                            # a1, log_prob1 = actor0.evaluate(b_s[:, state_number:state_number * 2])
                            # a = torch.hstack((a0, a1))
                            # # a = torch.cat((a0, a1),dim=1)
                            # log_prob = (log_prob0 + log_prob1) / 2
                            # # log_prob = torch.hstack((log_prob0.mean(axis=1).unsqueeze(dim=1), log_prob1.mean(axis=1).unsqueeze(dim=1)))
                            # q1, q2 = critic0.get_v(b_s, a)
                            # q = torch.min(q1, q2)
                            #
                            # actor_loss = (entroy0.alpha * log_prob - q).mean()
                            # alpha_loss = -(entroy0.log_alpha.exp() * (log_prob + entroy0.target_entropy).detach()).mean()
                            #
                            # actor0.learn(actor_loss)
                            # # actor1.learn(actor_loss)
                            # entroy0.learn(alpha_loss)
                            # entroy0.alpha = entroy0.log_alpha.exp()
                            # # 软更新
                            # critic0.soft_update()
                                # TODO 方法零
                                # if not done[i]:
                                new_action, log_prob_ = actors[i].evaluate(b_s_[:, state_number*i:state_number*(i+1)])
                                target_q1, target_q2 = critics[i].target_critic_v(b_s_, new_action)
                                target_q = b_r[:, i:(i+1)] + GAMMA * (torch.min(target_q1, target_q2) - entroys[i].alpha * log_prob_)
                                current_q1, current_q2 = critics[i].get_v(b_s, b_a[:, action_number*i:action_number*(i+1)])
                                critics[i].learn(current_q1, current_q2, target_q.detach())
                                a, log_prob = actors[i].evaluate(b_s[:, state_number*i:state_number*(i+1)])
                                q1, q2 = critics[i].get_v(b_s, a)
                                q = torch.min(q1, q2)
                                actor_loss = (entroys[i].alpha * log_prob - q).mean()
                                alpha_loss = -(entroys[i].log_alpha.exp() * (
                                                log_prob + entroys[i].target_entropy).detach()).mean()
                                actors[i].learn(actor_loss)
                                entroys[i].learn(alpha_loss)
                                entroys[i].alpha = entroys[i].log_alpha.exp()
                                # 软更新
                                critics[i].soft_update()
                                    # #TODO 方法一
                                    # new_action_0, log_prob_0 = actors[i].evaluate(b_s_[:, :state_number])
                                    # new_action_1, log_prob_1 = actors[i].evaluate(b_s_[:, state_number:state_number * 2])
                                    # new_action = torch.hstack((new_action_0, new_action_1))
                                    # # new_action = torch.cat((new_action_0, new_action_1),dim=1)
                                    # # log_prob_ = (log_prob_0 + log_prob_1) / 2
                                    # # log_prob_=torch.hstack((log_prob_0.mean(axis=1).unsqueeze(dim=1),log_prob_1.mean(axis=1).unsqueeze(dim=1)))
                                    # target_q1, target_q2 = critics[i].target_critic_v(b_s_, new_action)
                                    # if i==0:
                                    #     target_q = b_r[:, i:(i+1)] + GAMMA * (torch.min(target_q1, target_q2) - entroys[i].alpha * log_prob_0)
                                    # elif i==1:
                                    #     target_q = b_r[:, i:(i+1)] + GAMMA * (torch.min(target_q1, target_q2) - entroys[i].alpha * log_prob_1)
                                    # current_q1, current_q2 = critics[i].get_v(b_s, b_a)
                                    # critics[i].learn(current_q1, current_q2, target_q.detach())
                                    # a0, log_prob0 = actors[i].evaluate(b_s[:, :state_number])
                                    # a1, log_prob1 = actors[i].evaluate(b_s[:, state_number:state_number * 2])
                                    # a = torch.hstack((a0, a1))
                                    # # a = torch.cat((a0, a1),dim=1)
                                    # # log_prob = (log_prob0 + log_prob1) / 2
                                    # # log_prob = torch.hstack((log_prob0.mean(axis=1).unsqueeze(dim=1), log_prob1.mean(axis=1).unsqueeze(dim=1)))
                                    # q1, q2 = critics[i].get_v(b_s, a)
                                    # q = torch.min(q1, q2)
                                    # if i == 0:
                                    #     actor_loss = (entroys[i].alpha * log_prob0 - q).mean()
                                    #     alpha_loss = -(entroys[i].log_alpha.exp() * (log_prob0 + entroys[i].target_entropy).detach()).mean()
                                    # elif i == 1:
                                    #     actor_loss = (entroys[i].alpha * log_prob1 - q).mean()
                                    #     alpha_loss = -(entroys[i].log_alpha.exp() * (log_prob1 + entroys[i].target_entropy).detach()).mean()
                                    # actors[i].learn(actor_loss)
                                    # entroys[i].learn(alpha_loss)
                                    # entroys[i].alpha = entroys[i].log_alpha.exp()
                                    # # 软更新
                                    # critics[i].soft_update()
                                # #TODO 方法二
                                # new_action_0, log_prob_0 = actors[i].evaluate(b_s_[:, :state_number])
                                # new_action_1, log_prob_1 = actors[i].evaluate(b_s_[:, state_number:state_number * 2])
                                # new_action = torch.hstack((new_action_0, new_action_1))
                                # log_prob_ = (log_prob_0 + log_prob_1) / 2
                                # # log_prob_=torch.hstack((log_prob_0.mean(axis=1).unsqueeze(dim=1),log_prob_1.mean(axis=1).unsqueeze(dim=1)))
                                # target_q1, target_q2 = critics[i].target_critic_v(b_s_, new_action)
                                # target_q = b_r + GAMMA * (torch.min(target_q1, target_q2) - entroys[i].alpha * log_prob_)
                                # current_q1, current_q2 = critics[i].get_v(b_s, b_a)
                                # critics[i].learn(current_q1, current_q2, target_q.detach())
                                # a0, log_prob0 = actors[i].evaluate(b_s[:, :state_number])
                                # a1, log_prob1 = actors[i].evaluate(b_s[:, state_number:state_number * 2])
                                # a = torch.hstack((a0, a1))
                                # log_prob = (log_prob0 + log_prob1) / 2
                                # # log_prob = torch.hstack((log_prob0.mean(axis=1).unsqueeze(dim=1), log_prob1.mean(axis=1).unsqueeze(dim=1)))
                                # q1, q2 = critics[i].get_v(b_s, a)
                                # q = torch.min(q1, q2)
                                # actor_loss = ( entroys[i].alpha * log_prob - q).mean()
                                # actors[i].learn(actor_loss)
                                # alpha_loss = -( entroys[i].log_alpha.exp() * (log_prob +  entroys[i].target_entropy).detach()).mean()
                                # entroys[i].learn(alpha_loss)
                                # entroys[i].alpha =  entroys[i].log_alpha.exp()
                                # # 软更新
                                # critics[i].soft_update()
                            # new_action_0, log_prob_0 = actor.evaluate(b_s_[:, :state_number])
                            # new_action_1, log_prob_1 = actor.evaluate(b_s_[:, state_number:state_number*2])
                            # new_action=torch.hstack((new_action_0,new_action_1))
                            # log_prob_=(log_prob_0+log_prob_1)/2
                            # # log_prob_=torch.hstack((log_prob_0.mean(axis=1).unsqueeze(dim=1),log_prob_1.mean(axis=1).unsqueeze(dim=1)))
                            # target_q1,target_q2=critic.target_critic_v(b_s_,new_action)
                            # target_q=b_r+GAMMA*(torch.min(target_q1,target_q2)-entroy.alpha*log_prob_)
                            # current_q1, current_q2 = critic.get_v(b_s, b_a)
                            # critic.learn(current_q1,current_q2,target_q.detach())
                            # a0,log_prob0=actor.evaluate(b_s[:, :state_number])
                            # a1, log_prob1 = actor.evaluate(b_s[:, state_number:state_number*2])
                            # a = torch.hstack((a0, a1))
                            # log_prob=(log_prob0+log_prob1)/2
                            # # log_prob = torch.hstack((log_prob0.mean(axis=1).unsqueeze(dim=1), log_prob1.mean(axis=1).unsqueeze(dim=1)))
                            # q1,q2=critic.get_v(b_s,a)
                            # q=torch.min(q1,q2)
                            # actor_loss = (entroy.alpha * log_prob - q).mean()
                            # actor.learn(actor_loss)
                            # alpha_loss = -(entroy.log_alpha.exp() * (log_prob + entroy.target_entropy).detach()).mean()
                            # entroy.learn(alpha_loss)
                            # entroy.alpha=entroy.log_alpha.exp()
                            # # 软更新
                            # critic.soft_update()
                        observation = observation_
                        reward_totle += reward.mean()
                        reward_totle0 += float(reward[0])
                        reward_totle1 += float(reward[1])
                        if RENDER:
                            env.render()
                        if done:
                            break
                    print("Ep: {} rewards: {}".format(episode, reward_totle))
                    all_ep_r[k].append(reward_totle)
                    all_ep_r0[k].append(reward_totle0)
                    all_ep_r1[k].append(reward_totle1)
                    if episode % 20 == 0 and episode > 200:#保存神经网络参数
                        save_data = {'net': actors[0].action_net.state_dict(), 'opt': actors[0].optimizer.state_dict()}
                        torch.save(save_data, "G:\path planning\Path_SAC_actor_L1.pth")
                        save_data = {'net': actors[1].action_net.state_dict(), 'opt': actors[1].optimizer.state_dict()}
                        torch.save(save_data, "G:\path planning\Path_SAC_actor_F1.pth")
                # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
                # plt.xlabel('Episode')
                # plt.ylabel('Total reward')
                # plt.figure(2, figsize=(8, 4), dpi=150)
                # plt.plot(np.arange(len(all_ep_r0)), all_ep_r0)
                # plt.xlabel('Episode')
                # plt.ylabel('Leader reward')
                # plt.figure(3, figsize=(8, 4), dpi=150)
                # plt.plot(np.arange(len(all_ep_r1)), all_ep_r1)
                # plt.xlabel('Episode')
                # plt.ylabel('Follower reward')
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
            plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='MASAC', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Total reward')
            plt.figure(2, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_L_mean)), all_ep_L_mean, label='MASAC', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_L_mean)), all_ep_L_max, all_ep_L_min, alpha=0.6,
                             facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Leader reward')
            plt.figure(3, figsize=(8, 4), dpi=150)
            plt.margins(x=0)
            plt.plot(np.arange(len(all_ep_F_mean)), all_ep_F_mean, label='MASAC', color='#e75840')
            plt.fill_between(np.arange(len(all_ep_F_mean)), all_ep_F_max, all_ep_F_min, alpha=0.6,
                             facecolor='#e75840')
            plt.xlabel('Episode')
            plt.ylabel('Follower reward')
            plt.legend()
            plt.show()
            env.close()
    else:
        print('SAC测试中...')
        aa = Actor()
        checkpoint_aa = torch.load("G:\path planning\Path_SAC_actor_L1.pth")
        aa.action_net.load_state_dict(checkpoint_aa['net'])
        bb = Actor()
        checkpoint_bb = torch.load("G:\path planning\Path_SAC_actor_F1.pth")
        bb.action_net.load_state_dict(checkpoint_bb['net'])
        action = np.zeros((N_Agent+M_Enemy, action_number))
        win_times = 0
        average_FKR=0
        average_timestep=0
        average_integral_V=0
        average_integral_U= 0
        all_ep_V, all_ep_U, all_ep_T, all_ep_F = [], [], [], []
        for j in range(TEST_EPIOSDE):
            state = env.reset()
            total_rewards = 0
            integral_V=0
            integral_U=0
            v,v1,Dis=[],[],[]
            for timestep in range(EP_LEN):
                for i in range(N_Agent):
                    action[i] = aa.choose_action(state[i])
                for i in range(M_Enemy):
                    action[i+1] = bb.choose_action(state[i+1])
                # action[0] = aa.choose_action(state[0])
                # action[1] = bb.choose_action(state[1])
                new_state, reward,done,win,team_counter,dis = env.step(action)  # 执行动作
                if win:
                    win_times += 1
                v.append(state[0][2]*30)
                v1.append(state[1][2]*30)
                Dis.append(dis)
                integral_V+=state[0][2]
                integral_U+=abs(action[0]).sum()
                total_rewards += reward.mean()
                state = new_state
                if RENDER:
                    env.render()
                if done:
                    break
            FKR=team_counter/timestep
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
            # d = {"leader": v, "follower": v1 }
            # d = {"distance": Dis}
            # f = open(shoplistfile_test, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
            # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
            # f.close()
            # plt.plot(np.arange(len(v)), v)
            # plt.plot(np.arange(len(v1)), v1)
            # plt.plot(np.arange(len(Dis)), Dis)
            # plt.show()
        print('任务完成率',win_times / TEST_EPIOSDE)
        print('平均最大编队保持率', average_FKR/TEST_EPIOSDE)
        print('平均最短飞行时间', average_timestep/TEST_EPIOSDE)
        print('平均最短飞行路程', average_integral_V/TEST_EPIOSDE)
        print('平均最小能量损耗', average_integral_U/TEST_EPIOSDE)
        # d = {"all_ep_V": all_ep_V, "all_ep_U": all_ep_U, "all_ep_T": all_ep_T, "all_ep_F": all_ep_F, }
        # f = open(shoplistfile_test1, 'wb')  # 二进制打开，如果找不到该文件，则创建一个
        # pkl.dump(d, f, pkl.HIGHEST_PROTOCOL)  # 写入文件
        # f.close()
        env.close()

if __name__ == '__main__':
    main()