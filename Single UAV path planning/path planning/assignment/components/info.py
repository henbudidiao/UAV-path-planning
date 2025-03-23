# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:21
import pygame
from assignment import constants as C
pygame.font.init()

class Info():
    def __init__(self,state,game_info):#state是阶段，游戏的不同阶段要展示不同信息
        self.game_info=game_info
        self.state=state
        self.create_state_labels()#创建某阶段特有信息
        self.create_info_labels()#创建通用的信息

    def create_state_labels(self):
        self.state_labels=[]
        if self.state=='main_menu':
            self.state_labels.append((self.create_label('设置', size=30), (700, 0)))
            self.menu_info_rect=self.create_label('设置', size=30).get_rect()
            self.menu_info_rect.x=self.menu_info_rect.x+700
        elif self.state=='load_screen':
            self.state_labels.append((self.create_label('小亮出品 必属精品', size=60), (150, 0)))
        elif self.state=='battle_screen':
            self.state_labels.append((self.create_label('空战模拟界面', size=30), (300, 0)))
            self.state_labels.append((self.create_label('子弹数', size=15), (600, 0)))
            # self.state_labels.append((self.create_label('湍流龙击打', size=15), (300, 550)))
            # self.state_labels.append((self.create_label('克制', size=15), (420, 550)))
            # self.state_labels.append((self.create_label('巨龙之击', size=15), (540, 550)))
        elif self.state=='game_over':
            self.state_labels.append((self.create_label('Game Over', size=60,flag='E',color=C.RED), (200, 300)))
            self.state_labels.append((self.create_label('{}胜！'.format(self.game_info['win']), size=60,color=C.RED), (500, 305)))
            self.state_labels.append((self.create_label('当前是第{}场游戏'.format(self.game_info['epsoide']),size=30),(300,150)))
            self.state_labels.append((self.create_label('失败{}场'.format(self.game_info['enemy_win']),size=30),(300,190)))
            self.state_labels.append((self.create_label('成功{}场'.format(self.game_info['hero_win']), size=30), (300, 230)))

    def create_info_labels(self):
        self.info_labels=[]
        self.info_labels.append((self.create_label('通用信息',size=20),(0,0)))
        self.info_rect=self.create_label('通用信息',size=20).get_rect()
        # self.info_rect.x=self.info_rect.x+0
        # self.info_rect.y = self.info_rect.y + 0

    def create_label(self,label,size=40,flag='Chinese',color=C.WHITE):#把文字渲染为图片
        if flag=='Chinese':
            font=pygame.font.SysFont(C.FONT_CHINESE,size)
        else:
            font = pygame.font.SysFont(C.FONT_ENGLISH, size)
        label_image=font.render(label,1,color)
        return label_image

    def update(self,mouse_pos):
        if self.info_rect.collidepoint(mouse_pos):
            self.info_labels[0]=(self.create_label('通用信息',size=20,color=C.GREEN),(0,0))
            self.info_labels.append((self.create_label('当前是第{}场游戏'.format(self.game_info['epsoide']),size=20),(0,20)))
            self.info_labels.append((self.create_label('失败赢{}场'.format(self.game_info['enemy_win']), size=20), (0, 40)))
            self.info_labels.append((self.create_label('成功赢{}场'.format(self.game_info['hero_win']), size=20), (0, 60)))
        else:
            self.info_labels.clear()
            self.info_labels.append((self.create_label('通用信息', size=20), (0, 0)))
        if not C.OPEN_MENU and self.state=='main_menu':
            if self.menu_info_rect.collidepoint(mouse_pos):
                self.state_labels[0]=(self.create_label('设置', size=30,color=C.GREEN), (700, 0))
                if C.CLICK:
                    C.OPEN_MENU=True
            else:
                self.state_labels[0]=(self.create_label('设置', size=30), (700, 0))

    def draw(self,surface):
        for label in self.state_labels:
            surface.blit(label[0],label[1])
        for label in self.info_labels:
            surface.blit(label[0], label[1])