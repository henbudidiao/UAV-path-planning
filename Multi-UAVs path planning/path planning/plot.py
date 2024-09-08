# # -*- coding: utf-8 -*-
# #开发者：Bright Fang
# #开发时间：2023/9/11 17:27
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle as p
plt.rcParams['font.sans-serif'] = 'SIMSun'  # 设置字体为仿宋（FangSong）
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号’-'显示为方块的问题
'''训练曲线'''
shoplistfile_split = 'G:\path planning\MASAC_new' #保存文件数据所在文件的文件名
shoplistfile1 = 'G:\path planning\MADDPG'  #保存文件数据所在文件的文件名

f_split = open(shoplistfile_split, 'rb')
storedlist_split = p.load(f_split)
f_split1 = open(shoplistfile1, 'rb')
storedlist_split1 = p.load(f_split1)

all_ep_r_mean_split=storedlist_split['all_ep_r_mean']
all_ep_r_std_split=storedlist_split['all_ep_r_std']
all_ep_r_max_split = all_ep_r_mean_split + all_ep_r_std_split * 0.95
all_ep_r_min_split = all_ep_r_mean_split - all_ep_r_std_split * 0.95

all_ep_L_mean_split=storedlist_split['all_ep_L_mean']
all_ep_L_std_split=storedlist_split['all_ep_L_std']
all_ep_L_max_split = all_ep_L_mean_split + all_ep_L_std_split * 0.95
all_ep_L_min_split = all_ep_L_mean_split - all_ep_L_std_split * 0.95

all_ep_F_mean_split=storedlist_split['all_ep_F_mean']
all_ep_F_std_split=storedlist_split['all_ep_F_std']
all_ep_F_max_split = all_ep_F_mean_split + all_ep_F_std_split * 0.95
all_ep_F_min_split = all_ep_F_mean_split - all_ep_F_std_split * 0.95

all_ep_r_mean_split1=storedlist_split1['all_ep_r_mean']
all_ep_r_std_split1=storedlist_split1['all_ep_r_std']
all_ep_r_max_split1 = all_ep_r_mean_split1 + all_ep_r_std_split1 * 0.95
all_ep_r_min_split1 = all_ep_r_mean_split1 - all_ep_r_std_split1 * 0.95
# plt.figure(1, figsize=(8, 4), dpi=200)
# plt.plot(np.arange(len(all_ep_L_mean_split)), all_ep_L_mean_split, color='#115840',label='领导者UAV子奖励')
# plt.fill_between(np.arange(len(all_ep_L_mean_split)), all_ep_L_max_split, all_ep_L_min_split, alpha=0.3, facecolor='#115840')
# plt.legend()
# plt.xlabel('回合数')
# plt.ylabel('奖励值')
#
# plt.plot(np.arange(len(all_ep_F_mean_split)), all_ep_F_mean_split, color='#3299CC',label='跟随者UAV子奖励')
# plt.fill_between(np.arange(len(all_ep_F_mean_split)), all_ep_F_max_split, all_ep_F_min_split, alpha=0.3, facecolor='#3299CC')
# plt.legend()
# plt.xlabel('回合数')
# plt.ylabel('奖励值')

# plt.plot(np.arange(len(all_ep_r_mean_split1)), all_ep_r_mean_split1,label='MADDPG算法')
# plt.fill_between(np.arange(len(all_ep_r_mean_split1)), all_ep_r_max_split1, all_ep_r_min_split1, alpha=0.3)
# n = 20  # 每隔 20 个数取平均
# y = all_ep_r_mean_split1
# m = len(y)//n  # 计算可以分成几组
# averages = np.array([np.mean(y[i * n:(i + 1) * n]) for i in range(m)])
# x = np.arange(m) * n
# plt.plot(x, averages, 'b', linewidth=1.5)
# plt.legend()
# plt.xlabel('回合数')
# plt.ylabel('奖励值')
#
plt.plot(np.arange(len(all_ep_r_mean_split)), all_ep_r_mean_split, color='#e75840',label='MASAC算法')
plt.fill_between(np.arange(len(all_ep_r_mean_split)), all_ep_r_max_split, all_ep_r_min_split, alpha=0.3, facecolor='#e75840')
plt.legend()
# y1 = all_ep_r_mean_split
# averages1 = np.array([np.mean(y1[i * n:(i + 1) * n]) for i in range(m)])
# x1 = np.arange(m) * n
# plt.plot(x1, averages1, 'r', linewidth=1.5)
plt.xlabel('回合数')
plt.ylabel('奖励值')
#
plt.xlim(xmin=0,xmax=500)
plt.ylim(ymin=-4000,ymax=1500)
plt.savefig("奖励.eps", format='eps', dpi=1000)
plt.show()

'''协同测试'''
# shoplistfile_split = 'G:\path planning\MASAC_test' #保存文件数据所在文件的文件名
# shoplistfile_split1 = 'G:\path planning\MASAC_v_test'
# shoplistfile_split2 = 'G:\path planning\MASAC_d_test'
# f_split = open(shoplistfile_split, 'rb')
# storedlist_split = p.load(f_split)
#
# f_split1 = open(shoplistfile_split1, 'rb')
# storedlist_split1 = p.load(f_split1)
#
# f_split2 = open(shoplistfile_split2, 'rb')
# storedlist_split2 = p.load(f_split2)
#
# all_ep_L=storedlist_split['leader']
# all_ep_F=storedlist_split['follower']
#
# all_ep_L1=storedlist_split1['leader']
# all_ep_F1=storedlist_split1['follower']
#
# all_ep_D=storedlist_split2['distance']
# fig, ax=plt.subplots(1, figsize=(8, 4), dpi=200)
# plt.plot(np.arange(len(all_ep_L)), all_ep_L, label='领导者UAV航向角')
# plt.plot(np.arange(len(all_ep_F)), all_ep_F, label='跟随者UAV航向角',linestyle='--')
# plt.legend()
# plt.xlabel('时间')
# plt.ylabel('航向角')
# plt.grid(linestyle=":")
# axins = ax.inset_axes((0.3, 0.4, 0.6, 0.4))
# axins.plot(np.arange(len(all_ep_L)), all_ep_L)
# axins.plot(np.arange(len(all_ep_F)), all_ep_F,linestyle='--')
# # 调整子坐标系的显示范围
# axins.set_xlim(100, 130)
# axins.set_ylim(60, 100)
# mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='k', lw=1)
# plt.xlim(xmin=0)
# plt.ylim(ymin=0,ymax=360)
# plt.savefig("航向角.eps", format='eps', dpi=1000,bbox_inches = 'tight')

# fig, ax=plt.subplots(1, figsize=(8, 4), dpi=200)
# plt.plot(np.arange(len(all_ep_L1)), all_ep_L1, label='领导者UAV速度')
# plt.plot(np.arange(len(all_ep_F1)), all_ep_F1, label='跟随者UAV速度',linestyle='--')
# plt.legend()
# plt.xlabel('时间')
# plt.ylabel('速度')
# plt.grid(linestyle=":")
# axins = ax.inset_axes((0.3, 0.1, 0.6, 0.4))
# axins.plot(np.arange(len(all_ep_L1)), all_ep_L1)
# axins.plot(np.arange(len(all_ep_F1)), all_ep_F1,linestyle='--')
# # # 调整子坐标系的显示范围
# axins.set_xlim(180, 210)
# axins.set_ylim(18, 20)
# mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=1)
# plt.xlim(xmin=0)
# plt.ylim(ymin=0,ymax=30)
# plt.savefig("速度.eps", format='eps', dpi=1000,bbox_inches = 'tight')
#
# plt.figure(3, figsize=(8, 4), dpi=200)
# plt.plot(np.arange(len(all_ep_D)), all_ep_D,label='领导者与跟随者UAV的距离')
# plt.legend()
# plt.xlabel('时间')
# plt.ylabel('距离')
# plt.xlim(xmin=0)
# plt.ylim(ymin=0)
# plt.grid()
# plt.savefig("距离.eps", format='eps', dpi=1000,bbox_inches = 'tight')
# plt.show()
'''指标比较'''
# shoplistfile_split = 'G:\path planning\sample_compare' #保存文件数据所在文件的文件名
# shoplistfile_split1 = 'G:\path planning\MADDPG_compare' #保存文件数据所在文件的文件名
# shoplistfile_split2 = 'G:\path planning\MASAC_compare' #保存文件数据所在文件的文件名
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
# all_ep_F=storedlist_split['all_ep_F']
# all_ep_V1=storedlist_split1['all_ep_V']
# all_ep_U1=storedlist_split1['all_ep_U']
# all_ep_T1=storedlist_split1['all_ep_T']
# all_ep_F1=storedlist_split1['all_ep_F']
# all_ep_V2=storedlist_split2['all_ep_V']
# all_ep_U2=storedlist_split2['all_ep_U']
# all_ep_T2=storedlist_split2['all_ep_T']
# all_ep_F2=storedlist_split2['all_ep_F']
#
# plt.figure(1, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_V)), all_ep_V, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_V1)), all_ep_V1, label='MADDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_V2)), all_ep_V2, label='MASAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('飞行轨迹')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")
#
# plt.figure(2, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_U)), all_ep_U, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_U1)), all_ep_U1, label='MADDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_U2)), all_ep_U2, label='MASAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('能量损耗')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")
#
# plt.figure(3, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_T)), all_ep_T, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_T1)), all_ep_T1, label='MADDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_T2)), all_ep_T2, label='MASAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('飞行时间')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")
#
# plt.figure(4, figsize=(8, 4), dpi=150)
# plt.xlim(xmin=0, xmax=100)
# plt.plot(np.arange(len(all_ep_F)), all_ep_F, label='随机策略',  marker='^',color='k')
# plt.plot(np.arange(len(all_ep_F1)), all_ep_F1, label='MADDPG算法', marker='o',color='#3299CC')
# plt.plot(np.arange(len(all_ep_F2)), all_ep_F2, label='MASAC算法',  marker='*',color='#e75840')
# plt.xlabel('Monte Carlo测试回合数')
# plt.ylabel('编队保持率')
# plt.grid(linestyle=":")
# plt.legend(loc ="upper right")
# # plt.savefig("output.eps", format='eps', dpi=1000)
# plt.show()