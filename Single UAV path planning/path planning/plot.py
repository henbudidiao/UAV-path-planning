# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/9/10 15:32
import matplotlib.pyplot as plt
import numpy as np
import pickle as p
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
plt.rcParams['font.sans-serif'] = 'SimSun'  # 设置字体为仿宋（FangSong）
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号’-'显示为方块的问题
shoplistfile = 'E:\path planning\SAC' #保存文件数据所在文件的文件名
shoplistfile1 = 'E:\path planning\DDPG1' #保存文件数据所在文件的文件名
shoplistfile2 = 'E:\path planning\sample_test' #保存文件数据所在文件的文件名
f = open(shoplistfile, 'rb')
storedlist = p.load(f)
f = open(shoplistfile1, 'rb')
storedlist1 = p.load(f)
f = open(shoplistfile2, 'rb')
storedlist2 = p.load(f)

all_ep_r_mean=storedlist['all_ep_r_mean']
all_ep_r_std=storedlist['all_ep_r_std']
all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
all_ep_edge_mean=storedlist['all_ep_edge_mean']
all_ep_edge_std=storedlist['all_ep_edge_std']
all_ep_edge_max = all_ep_edge_mean + all_ep_edge_std * 0.95
all_ep_edge_min = all_ep_edge_mean - all_ep_edge_std * 0.95
all_ep_obstacle_mean=storedlist['all_ep_obstacle_mean']
all_ep_obstacle_std=storedlist['all_ep_obstacle_std']
all_ep_obstacle_max = all_ep_obstacle_mean + all_ep_obstacle_std * 0.95
all_ep_obstacle_min = all_ep_obstacle_mean - all_ep_obstacle_std * 0.95
all_ep_goal_mean=storedlist['all_ep_goal_mean']
all_ep_goal_std=storedlist['all_ep_goal_std']
all_ep_goal_max = all_ep_goal_mean + all_ep_goal_std * 0.95
all_ep_goal_min = all_ep_goal_mean - all_ep_goal_std * 0.95

all_ep_r_mean1=storedlist1['all_ep_r_mean']
all_ep_r_std1=storedlist1['all_ep_r_std']
all_ep_r_max1 = all_ep_r_mean1 + all_ep_r_std1 * 0.95
all_ep_r_min1 = all_ep_r_mean1 - all_ep_r_std1 * 0.95
all_ep_edge_mean1=storedlist1['all_ep_edge_mean']
all_ep_edge_std1=storedlist1['all_ep_edge_std']
all_ep_edge_max1 = all_ep_edge_mean1 + all_ep_edge_std1 * 0.95
all_ep_edge_min1 = all_ep_edge_mean1 - all_ep_edge_std1 * 0.95
all_ep_goal_mean1=storedlist1['all_ep_goal_mean']
all_ep_goal_std1=storedlist1['all_ep_goal_std']
all_ep_goal_max1 = all_ep_goal_mean1 + all_ep_goal_std1 * 0.95
all_ep_goal_min1 = all_ep_goal_mean1 - all_ep_goal_std1 * 0.95
# plt.figure(1, figsize=(8, 4), dpi=150)
# # plt.margins(x=0)
# plt.ylim(ymin=-2500,ymax=500)
# plt.xlim(xmin=0,xmax=500)
# plt.plot(np.arange(len(all_ep_edge_mean1)), all_ep_edge_mean1, label='DDPG算法',alpha=0.5)
# # plt.fill_between(np.arange(len(all_ep_edge_mean1)), all_ep_edge_max1, all_ep_edge_min1, alpha=0.5,facecolor='#3299CC')
# n = 20  # 每隔 20 个数取平均
# y1 = all_ep_edge_mean1
# m = len(y1)//n  # 计算可以分成几组
# averages1 = np.array([np.mean(y1[i * n:(i + 1) * n]) for i in range(m)])
# x1 = np.arange(m) * n
# plt.plot(x1, averages1, 'b', linewidth=1.5)
# plt.plot(np.arange(len(all_ep_edge_mean)), all_ep_edge_mean, label='SAC算法', color="#e75840",alpha=0.5)
# # plt.fill_between(np.arange(len(all_ep_edge_mean)), all_ep_edge_max, all_ep_edge_min, alpha=0.6,
# #                  facecolor='#e75840')
# plt.legend()
# n = 20  # 每隔 20 个数取平均
# y= all_ep_edge_mean
# m = len(y)//n  # 计算可以分成几组
# averages = np.array([np.mean(y[i * n:(i + 1) * n]) for i in range(m)])
# x = np.arange(m) * n
# plt.plot(x, averages, 'r', linewidth=1.5)
# plt.xlabel('回合数')
# plt.ylabel('奖励值')
#
# plt.figure(2, figsize=(8, 4), dpi=150)
# plt.ylim(ymin=-2000,ymax=2000)
# plt.xlim(xmin=0,xmax=500)
# plt.plot(np.arange(len(all_ep_goal_mean1)), all_ep_goal_mean1, label='DDPG算法',alpha=0.5)
# # plt.fill_between(np.arange(len(all_ep_goal_mean1)), all_ep_goal_max1, all_ep_goal_min1, alpha=0.5,facecolor='#3299CC')
# n = 20  # 每隔 20 个数取平均
# y1 = all_ep_goal_mean1
# m = len(y1)//n  # 计算可以分成几组
# averages1 = np.array([np.mean(y1[i * n:(i + 1) * n]) for i in range(m)])
# x1 = np.arange(m) * n
# plt.plot(x1, averages1, 'b', linewidth=1.5)
# plt.plot(np.arange(len(all_ep_goal_mean)), all_ep_goal_mean, label='SAC算法',color="#e75840",alpha=0.5 )
# # plt.fill_between(np.arange(len(all_ep_goal_mean)), all_ep_goal_max, all_ep_goal_min, alpha=0.6,
# #                  facecolor='#e75840')
# plt.legend()
# n = 20  # 每隔 20 个数取平均
# y= all_ep_goal_mean
# m = len(y)//n  # 计算可以分成几组
# averages = np.array([np.mean(y[i * n:(i + 1) * n]) for i in range(m)])
# x = np.arange(m) * n
# plt.plot(x, averages, 'r', linewidth=1.5)
# plt.xlabel('回合数')
# plt.ylabel('奖励值')
#
# plt.figure(3, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=500)
# plt.ylim(ymin=-3000, ymax=2000)
# plt.plot(np.arange(len(all_ep_r_mean1)), all_ep_r_mean1, label='DDPG算法')
# # plt.fill_between(np.arange(len(all_ep_r_mean1)), all_ep_r_max1, all_ep_r_min1, alpha=0.5,facecolor='#3299CC')
# n = 20  # 每隔 20 个数取平均
# y1 = all_ep_r_mean1
# m = len(y1)//n  # 计算可以分成几组
# averages1 = np.array([np.mean(y1[i * n:(i + 1) * n]) for i in range(m)])
# x1 = np.arange(m) * n
# plt.plot(x1, averages1, 'b', linewidth=1.5)
#
# plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='SAC算法',color='#e75840')
# # plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.5,facecolor='#e75840')
# plt.xlabel('回合数')
# plt.ylabel('奖励值')
# y = all_ep_r_mean
# averages = np.array([np.mean(y[i * n:(i + 1) * n]) for i in range(m)])
# x = np.arange(m) * n
# plt.plot(x, averages, 'r', linewidth=1.5)
# plt.legend()
# plt.show()

'''测试总奖励比较'''
# all_ep_r_mean = storedlist['all_ep_r_mean']
# all_ep_r_std = storedlist['all_ep_r_std']
# all_ep_r_max = all_ep_r_mean + all_ep_r_std * 0.95
# all_ep_r_min = all_ep_r_mean - all_ep_r_std * 0.95
#
# all_ep_r_mean1 = storedlist1['all_ep_r_mean']
# all_ep_r_std1 = storedlist1['all_ep_r_std']
# all_ep_r_max1 = all_ep_r_mean1 + all_ep_r_std1 * 0.95
# all_ep_r_min1 = all_ep_r_mean1 - all_ep_r_std1 * 0.95
#
# all_ep_r_mean2 = storedlist2['all_ep_r_mean']
# all_ep_r_std2 = storedlist2['all_ep_r_std']
# all_ep_r_max2 = all_ep_r_mean2 + all_ep_r_std2 * 0.95
# all_ep_r_min2 = all_ep_r_mean2 - all_ep_r_std2* 0.95
# # plt.figure(1, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_r_mean2)), all_ep_r_mean2, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_r_mean1)), all_ep_r_mean1, label='DDPG算法', marker='o',color='#3299CC')
# # plt.fill_between(np.arange(len(all_ep_r_mean1)), all_ep_r_max1, all_ep_r_min1, alpha=0.6, facecolor='#3299CC')
# plt.plot(np.arange(len(all_ep_r_mean)), all_ep_r_mean, label='SAC算法',  marker='*',color='#e75840')
# # plt.fill_between(np.arange(len(all_ep_r_mean)), all_ep_r_max, all_ep_r_min, alpha=0.6, facecolor='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('总奖励')
# plt.grid()
# plt.legend()
# plt.savefig("output.eps", format='eps', dpi=1000)
# plt.show()
'''测试指标比较'''
# shoplistfile_split = 'E:\path planning\sample_indextest' #保存文件数据所在文件的文件名
# shoplistfile_split1 = 'E:\path planning\DDPG_indextest' #保存文件数据所在文件的文件名
# shoplistfile_split2 = 'E:\path planning\SAC_indextest' #保存文件数据所在文件的文件名
#
# f_split = open(shoplistfile_split, 'rb')
# storedlist_split = p.load(f_split)
# f_split = open(shoplistfile_split1, 'rb')
# storedlist_split1 = p.load(f_split)
# f_split = open(shoplistfile_split2, 'rb')
# storedlist_split2 = p.load(f_split)
#
# all_ep_V=storedlist_split['all_ep_V']
# all_ep_U=storedlist_split['all_ep_U']
# all_ep_T=storedlist_split['all_ep_T']
# all_ep_V1=storedlist_split1['all_ep_V']
# all_ep_U1=storedlist_split1['all_ep_U']
# all_ep_T1=storedlist_split1['all_ep_T']
# all_ep_V2=storedlist_split2['all_ep_V']
# all_ep_U2=storedlist_split2['all_ep_U']
# all_ep_T2=storedlist_split2['all_ep_T']

# plt.figure(1, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_V)), all_ep_V, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_V1)), all_ep_V1, label='DDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_V2)), all_ep_V2, label='SAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('飞行轨迹')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")

# plt.figure(2, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_U)), all_ep_U, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_U1)), all_ep_U1, label='DDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_U2)), all_ep_U2, label='SAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('能量损耗')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")

# plt.figure(3, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_T)), all_ep_T, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_T1)), all_ep_T1, label='DDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_T2)), all_ep_T2, label='SAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('飞行时间')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")
# plt.show()

'''训练的奖励曲线'''
plt.figure(1, figsize=(8, 4), dpi=150)
# #总奖励
# all_ep_r=[]
# all_ep_r1=[]
# for i in range(500):
#     if i == 0:
#         all_ep_r.append(all_ep_r_mean[i])
#         all_ep_r1.append(all_ep_r_mean1[i])
#     else:
#         all_ep_r.append(all_ep_r[ - 1] * 0.9 + all_ep_r_mean[i] * 0.1)
#         all_ep_r1.append(all_ep_r1[-1] * 0.9 + all_ep_r_mean1[i] * 0.1)
# plt.plot(np.arange(len(all_ep_r_mean1)), all_ep_r_mean1, label='DDPG算法')
# plt.plot(np.arange(len(all_ep_r)), all_ep_r, label='SAC算法',color='#e75840')
#边界子奖励
# all_ep_r=[]
# all_ep_r1=[]
# for i in range(500):
#     if i == 0:
#         all_ep_r.append(all_ep_edge_mean[i])
#         all_ep_r1.append(all_ep_edge_mean1[i])
#     else:
#         all_ep_r.append(all_ep_r[ - 1] * 0.9 + all_ep_edge_mean[i] * 0.1)
#         all_ep_r1.append(all_ep_r1[-1] * 0.9 + all_ep_edge_mean1[i] * 0.1)
# plt.plot(np.arange(len(all_ep_r1)), all_ep_r1, label='DDPG算法')
# plt.plot(np.arange(len(all_ep_r)), all_ep_r, label='SAC算法',color='#e75840')
#目标子奖励
# all_ep_r=[]
# all_ep_r1=[]
# for i in range(500):
#     if i == 0:
#         all_ep_r.append(all_ep_goal_mean[i])
#         all_ep_r1.append(all_ep_goal_mean1[i])
#     else:
#         all_ep_r.append(all_ep_r[ - 1] * 0.9 + all_ep_goal_mean[i] * 0.1)
#         all_ep_r1.append(all_ep_r1[-1] * 0.9 + all_ep_goal_mean1[i] * 0.1)
# plt.plot(np.arange(len(all_ep_r1)), all_ep_r1, label='DDPG算法')
# plt.plot(np.arange(len(all_ep_r)), all_ep_r, label='SAC算法',color='#e75840')
#
# plt.xlabel('回合数')
# plt.ylabel('奖励值')
# plt.xlim(xmin=0, xmax=500)
# plt.legend(loc ="upper right")
# plt.show()