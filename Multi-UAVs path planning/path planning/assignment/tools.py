# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:23
import pygame
from assignment import constants as C
import os

def load_graphics(path,accept=('.jpg','.png','.bmp','.gif')):#加载文件夹下的所有图片，放在集合graphics里
    graphics={}
    for pic in os.listdir(path):#从文件夹中遍历文件
        name,ext=os.path.splitext(pic)#把文件拆成 文件名+后缀
        if ext.lower() in accept:
            img=pygame.image.load(os.path.join(path,pic))
            if img.get_alpha():
                img=img.convert_alpha()
            else:
                img=img.convert()
            graphics[name]=img
    return graphics

def load_sound(path,accept=('.wav','.mp3')):#加载文件夹下的所有图片，放在集合graphics里
    sound={}
    for pic in os.listdir(path):#从文件夹中遍历文件
        name,ext=os.path.splitext(pic)#把文件拆成 文件名+后缀
        if ext.lower() in accept:
            sou=pygame.mixer.Sound(os.path.join(path,pic))
            sound[name]=sou
    return sound

# class Game():
#     def __init__(self,state_dict,start_state):
#         self.screen=pygame.display.get_surface()
#         self.clock=pygame.time.Clock()
#         self.state_dict=state_dict
#         self.state=self.state_dict[start_state]
#
#     def update(self,action):#场景阶段切换时的代码
#         if self.state.finished:
#             game_info=self.state.game_info
#             next_state=self.state.next
#             self.state.finished=False
#             # self.state=self.state_dict[next_state]#只有战斗界面
#             self.state.start(game_info)
#         return self.state.update(self.screen,action)#调用主菜单自己的 update方法，即自己的画面显示方式
    # def run(self):
    #     self.mouse_pos = (0,0)# 一定要初始化mouse_pos与mouse，不然在某些情况下用户启动程序会有bug
    #     self.mouse,self.mouse_his=0,0
    #     while True:
    #         self.mouse_his = self.mouse
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 pygame.display.quit()
    #                 quit()
    #             elif event.type == pygame.KEYDOWN:
    #                 self.keys = pygame.key.get_pressed()
    #             elif event.type==pygame.MOUSEMOTION:
    #                 self.mouse_pos=pygame.mouse.get_pos()
    #                 # print(self.mouse_pos)
    #             elif event.type==pygame.MOUSEBUTTONDOWN:
    #                 if event.button:
    #                     self.mouse=1
    #             elif event.type==pygame.MOUSEBUTTONUP:
    #                 if event.button:
    #                     self.mouse = 0
    #             elif event.type==C.CREATE_ENEMY_EVENT:
    #                 C.ENEMY_FLAG=True
    #             # print(self.mouse_his,self.mouse)
    #             if self.mouse_his and not self.mouse:
    #                 C.CLICK=True
    #             else:
    #                 C.CLICK=False
    #         # self.screen.fill((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
    #         # image=get_image(graphics['instinct'],(0,0,0),1)
    #         # self.screen.blit(image,(100,100))
    #         self.update()
    #         pygame.display.update()
    #         self.clock.tick(C.FPS)