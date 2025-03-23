# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2023/7/20 23:30
import numpy as np
import copy
import gym
from assignment import constants as C
from gym import spaces
import math
import random
import pygame
from assignment.components import player
from assignment import tools
from assignment.components import info
class RlGame(gym.Env):
    def __init__(self, n,m,l,render=False):
        self.hero_num = n
        self.enemy_num = m
        self.obstacle_num=l
        self.goal_num=1
        self.Render=render
        self.game_info = {
            'epsoide': 0,
            'hero_win': 0,
            'enemy_win': 0,
            'win': '未知',
        }
        if self.Render:
            pygame.init()
            pygame.mixer.init()
            self.SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

            pygame.display.set_caption("基于深度强化学习的空战场景无人机路径规划软件")

            self.GRAPHICS = tools.load_graphics('E:\path planning/assignment\source\image')

            self.SOUND = tools.load_sound('E:\path planning/assignment\source\music')
            self.clock = pygame.time.Clock()
            self.mouse_pos=(100,100)
            pygame.time.set_timer(C.CREATE_ENEMY_EVENT, C.ENEMY_MAKE_TIME)
            # self.res, init_extra, update_extra, skip_override, waypoints = simulate(filename='')

        # else:
        #     self.dispaly=None
        low = np.array([-1,-1])
        high=np.array([1,1])
        # self.action_space =spaces.Discrete(21)
        # self.action_space = spaces.Discrete(2)
        self.action_space=spaces.Box(low=low,high=high,dtype=np.float32)
        # self.action_space = [spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),
        #                      spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),spaces.Discrete(2),]
    def start(self):
        # self.game_info=game_info
        self.finished=False
        # self.next='game_over'
        self.set_battle_background()#战斗的背景
        self.set_enemy_image()
        self.set_hero_image()
        self.set_obstacle_image()
        self.set_goal_image()
        self.info = info.Info('battle_screen',self.game_info)
        # self.state = 'battle'
        self.counter=0
        self.counter_1 = 0
        self.counter_hero = 0
        self.enemy_counter=0
        self.enemy_counter_1 = 0
        #又定义了一个参数，为了放在start函数里重置
        self.enemy_num_start=self.enemy_num
        self.trajectory_x,self.trajectory_y=[],[]
        self.enemy_trajectory_x,self.enemy_trajectory_y=[[] for i in range(self.enemy_num)],[[] for i in range(self.enemy_num)]
        # RL状态
        # self.hero_state = np.zeros((self.hero_num, 4))
        # self.hero_α = np.zeros((self.hero_num, 1))
        self.uav_obs_check= np.zeros((self.hero_num, 1))

    def set_battle_background(self):
        self.battle_background = self.GRAPHICS['background']
        self.battle_background = pygame.transform.scale(self.battle_background,C.SCREEN_SIZE)  # 缩放
        self.view = self.SCREEN.get_rect()
        #若要移动的背景图像，请用下面的代码替换
        # bg1=player.BackgroundSprite(image_name='background3',size=C.SCREEN_SIZE)
        # bg2=player.BackgroundSprite(image_name='background3',size=C.SCREEN_SIZE)
        # bg2.rect.y=-bg2.rect.height
        # self.background_group=pygame.sprite.Group(bg1,bg2)

    def set_hero_image(self):
        self.hero = self.__dict__
        self.hero_group = pygame.sprite.Group()
        self.hero_image = self.GRAPHICS['fighter-blue']
        for i in range(self.hero_num):
            self.hero['hero'+str(i)]=player.Hero(image=self.hero_image)
            self.hero_group.add(self.hero['hero'+str(i)])

    def set_hero(self):
        self.hero = self.__dict__
        self.hero_group = pygame.sprite.Group()
        for i in range(self.hero_num):
            self.hero['hero'+str(i)]=player.Hero()
            self.hero_group.add(self.hero['hero'+str(i)])

    def set_enemy_image(self):
        self.enemy = self.__dict__
        self.enemy_group = pygame.sprite.Group()
        self.enemy_image = self.GRAPHICS['fighter-red']
        for i in range(self.enemy_num):
            self.enemy['enemy'+str(i)]=player.Enemy(image=self.enemy_image)
            self.enemy_group.add(self.enemy['enemy'+str(i)])

    def set_enemy(self):
        self.enemy = self.__dict__
        self.enemy_group = pygame.sprite.Group()
        for i in range(self.enemy_num):
            self.enemy['enemy'+str(i)]=player.Enemy()
            self.enemy_group.add(self.enemy['enemy'+str(i)])

    def set_obstacle_image(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        self.obstacle_image = self.GRAPHICS['hole']
        for i in range(self.obstacle_num):
            self.obstacle['obstacle'+str(i)]=player.Obstacle(image=self.obstacle_image)
            self.obstacle_group.add(self.obstacle['obstacle'+str(i)])

    def set_obstacle(self):
        self.obstacle = self.__dict__
        self.obstacle_group = pygame.sprite.Group()
        for i in range(self.obstacle_num):
            self.obstacle['obstacle'+str(i)]=player.Obstacle()
            self.obstacle_group.add(self.obstacle['obstacle'+str(i)])

    def set_goal_image(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        self.goal_image = self.GRAPHICS['goal']
        for i in range(self.goal_num):
            self.goal['goal'+str(i)]=player.Goal(image=self.goal_image)
            self.goal_group.add(self.goal['goal'+str(i)])

    def set_goal(self):
        self.goal = self.__dict__
        self.goal_group = pygame.sprite.Group()
        for i in range(self.goal_num):
            self.goal['goal'+str(i)]=player.Goal()
            self.goal_group.add(self.goal['goal'+str(i)])

    def update_game_info(self):#死亡后重置数据
        self.game_info['epsoide'] += 1
        self.game_info['enemy_win'] = self.game_info['epsoide'] - self.game_info['hero_win']

    def reset(self):#reset的仅是环境状态，
        # obs=np.zeros((self.n, 4))#这是个二维矩阵，n*2维,现在只考虑一个己方无人机，所以现在是一个一维的
        # game_info=self.my_game.state.game_info
        # self.my_game.state.start(game_info)
        if self.Render:
            self.start()
        else:
            self.set_hero()
            self.set_enemy()
            self.set_goal()
            self.set_obstacle()
        self.done = False
        self.hero_state = np.zeros((self.hero_num,7))
        self.hero_α = np.zeros((self.hero_num, 1))
        # self.goal_x,self.goal_y=random.randint(100, 500), random.randint(100, 200)
        return np.array([self.hero0.init_x/1000,self.hero0.init_y/1000,self.hero0.speed/20,self.hero0.theta*57.3/360
                            ,self.goal0.init_x/1000, self.goal0.init_y/1000,0
                         ])#np.array([self.my_game.state.hero['hero0'].posx/1000,self.my_game.state.hero['hero0'].posy/1000,self.my_game.state.hero['hero0'].speed/2,self.my_game.state.hero['hero0'].theta*57.3/360])#np.zeros((self.n,2)).flatten()

    def step(self,action):
        r = 0
        g_x=0
        g_y=0
        #空气阻力系数
        F_k=0.08
        #无人机质量,100是像素与现实速度的比例，因为10像素/帧对应现实的100m/s
        m=12000/100
        #扰动的加速度
        F_a=0
        # print(self.goal0.init_x)
        edge_r=0
        obstacle_r=0
        o_flag=0
        goal_r=0
        dis_1_obs = np.zeros((self.obstacle_num, 1))
        obstacle_r1 = np.zeros((self.enemy_num, 1))
        for i in range(self.hero_num):
            #空气阻力
            # self.hero['hero' + str(i)].F=F_k*math.pow(self.hero['hero' + str(i)].speed,2)
            # F_a=(self.hero['hero' + str(i)].F/m)*math.cos(self.hero['hero' + str(i)].theta * 57.3)
            # 己方与障碍物的碰撞检测
            # self.hero['hero' + str(i)].enemies = pygame.sprite.spritecollide(self.hero['hero' + str(i)],
            #                                                                      self.obstacle_group, False)
            dis_1_goal = math.hypot(self.hero['hero' + str(i)].posx - self.goal0.init_x, self.hero['hero' + str(i)].posy - self.goal0.init_y)
            for j in range(self.obstacle_num):
                dis_1_obs[j] = math.hypot(self.hero['hero' + str(i)].posx - self.obstacle['obstacle'+str(j)].init_x,
                                        self.hero['hero' + str(i)].posy - self.obstacle['obstacle'+str(j)].init_y)
            for j in range(self.enemy_num):
                obstacle_r1[j] = math.hypot(self.hero['hero' + str(i)].posx - self.enemy['enemy' + str(j)].posx,
                                        self.hero['hero' + str(i)].posy - self.enemy['enemy' + str(j)].posy)
            if self.hero['hero' + str(i)].posx <= C.ENEMY_AREA_X+50:
                edge_r=-2
            elif self.hero['hero' + str(i)].posx >= C.ENEMY_AREA_WITH:
                edge_r=-2
            if self.hero['hero' + str(i)].posy >= C.ENEMY_AREA_HEIGHT:
                edge_r=-2
            elif self.hero['hero' + str(i)].posy <= C.ENEMY_AREA_Y+50:
                edge_r=-2
            if dis_1_goal < 40 and not self.hero['hero' + str(i)].dead:
                goal_r = 1000.0
                self.hero['hero' + str(i)].die()
                self.hero['hero' + str(i)].win = True
                self.done= True
                self.game_info['hero_win'] += 1
                self.update_game_info()
                print('aa')
            # elif dis_1_goal < 100:
            #     r = 1.0
            elif (dis_1_obs[0] < 20 )and not self.hero['hero' + str(i)].dead:
                o_flag = 1
                obstacle_r = -500
                self.hero['hero' + str(i)].die()
                self.hero['hero' + str(i)].win = False
                self.done = True
                self.update_game_info()
                print('gg')
            elif (dis_1_obs[0] < 40  ) and not self.hero['hero' + str(i)].dead:
                o_flag = 1
                obstacle_r = -2
            elif (obstacle_r1[0]< 10 ):
                o_flag = 1
                obstacle_r = -500
                self.hero['hero' + str(i)].die()
                self.hero['hero' + str(i)].win = False
                self.done = True
                self.update_game_info()
                print('gg')
            elif (obstacle_r1[0]< 50 ):
                o_flag = 1
                obstacle_r = -2
            elif not self.hero['hero' + str(i)].dead:
                goal_r = -0.001 * dis_1_goal
            r=edge_r+obstacle_r+goal_r
            # init_to_goal=math.atan2((-150+self.hero['hero'+str(i)].init_y),(200-self.hero['hero'+str(i)].init_x))
            # uav_to_goal = math.atan2((-self.goal0.init_y + self.hero['hero' + str(i)].posy), (self.goal0.init_x - self.hero['hero' + str(i)].posx))
            # uav_to_obstacle = math.atan2((-self.obstacle0.init_y + self.hero['hero' + str(i)].posy), (self.obstacle0.init_x - self.hero['hero' + str(i)].posx))
            # self.hero_α[i] =  0.1*abs(uav_to_obstacle - self.hero['hero' + str(i)].theta)
            # r = r + self.hero_α[i]
            # print(self.hero_α[i],uav_to_goal,self.hero['hero' + str(i)].theta)
            # if len(self.hero['hero' + str(i)].enemies) > 0:
            #     r = -2000
            #     self.hero['hero' + str(i)].win = False
            #     self.done = True
            #     self.update_game_info()
            #     print('gg')
            self.hero_state[i] = [self.hero['hero' + str(i)].posx / 1000, self.hero['hero' + str(i)].posy / 1000,
                              self.hero['hero' + str(i)].speed / 20, self.hero['hero' + str(i)].theta * 57.3 / 360,
                              self.goal0.init_x / 1000, self.goal0.init_y / 1000, o_flag]
            # print('自己_{}状态：'.format(i),self.hero['hero'+str(i)].posx,self.hero['hero'+str(i)].posy)
            self.trajectory_x.append(self.hero['hero' + str(i)].posx)
            self.trajectory_y.append(self.hero['hero' + str(i)].posy)
        for i in range(self.enemy_num):
            self.enemy_trajectory_x[i].append(self.enemy['enemy' + str(i)].posx)
            self.enemy_trajectory_y[i].append(self.enemy['enemy' + str(i)].posy)
        # 自己更新位置
        self.hero_group.update(action,self.Render)
        self.enemy_group.update([random.uniform(-0.5,1), random.uniform(-0.5,1)],self.Render)
        hero_state = copy.deepcopy(self.hero_state)
        return hero_state.flatten(),r,self.done,edge_r,obstacle_r,goal_r,self.hero['hero0'].win
    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            elif event.type == pygame.MOUSEMOTION:
                self.mouse_pos = pygame.mouse.get_pos()
            elif event.type == C.CREATE_ENEMY_EVENT:
                C.ENEMY_FLAG = True
        # 画背景
        self.SCREEN.blit(self.battle_background, self.view)
        # 文字显示
        self.info.update(self.mouse_pos)
        # 画图
        self.draw(self.SCREEN)
        pygame.display.update()
        self.clock.tick(C.FPS)
    def draw(self,surface):
        # self.background_group.draw(surface)
        #敌占区的矩形
        pygame.draw.rect(surface, C.BLACK, C.ENEMY_AREA, 3)
        #目标星星
        # pygame.draw.polygon(surface, C.GREEN,[(200, 135), (205, 145), (215, 145), (210, 155), (213, 165), (200, 160), (187, 165), (190, 155), (185, 145), (195, 145)])
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 1)
        pygame.draw.circle(surface, C.RED, (self.goal0.init_x, self.goal0.init_y), 40,1)
        # pygame.draw.circle(surface, C.GREEN, (self.goal0.init_x, self.goal0.init_y),100, 1)
        # pygame.draw.circle(surface, C.BLACK, (self.obstacle0.init_x, self.obstacle0.init_y), 20, 1)
        # 画轨迹
        for i in range(1, len(self.trajectory_x)):
            pygame.draw.line(surface, C.BLUE, (self.trajectory_x[i - 1], self.trajectory_y[i - 1]), (self.trajectory_x[i], self.trajectory_y[i]))
        for j in range(self.enemy_num):
            for i in range(1, len(self.trajectory_x)):
                pygame.draw.line(surface, C.RED, (self.enemy_trajectory_x[j][i - 1], self.enemy_trajectory_y[j][i - 1]),
                                 (self.enemy_trajectory_x[j][i], self.enemy_trajectory_y[j][i]))
        #障碍物
        # pygame.draw.circle(surface, C.BLACK, (250, 300), 20)
        # 画自己
        self.hero_group.draw(surface)
        self.enemy_group.draw(surface)
        #障碍物
        self.obstacle_group.draw(surface)
        # 目标星星
        self.goal_group.draw(surface)
        #画文字信息
        self.info.draw(surface)
    def close(self):
        pygame.display.quit()
        quit()

