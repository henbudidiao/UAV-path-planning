# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 23:18
'''说明：目前程序每更新一次，相当于现实中过去1秒，即δt=1'''
import numpy as np
import math
import random
import pygame
# from assignment import set_up
from assignment import constants as C
from assignment.components import info
from assignment import tools
# class MySurface(pygame.Surface):
#     def  __init__(self):
#         super(MySurface, self).__init__()
#
#     def get_rect(self):
#         super().get_rect()
#
#         # print('mmmm')
dt=0.1
class GameSprite(pygame.sprite.Sprite):
    def __init__(self,image_name=None,size=(20,20),speed=1):
        super(GameSprite, self).__init__()
        # self.image=GRAPHICS[image_name]
        # self.image = pygame.transform.scale(self.image, size)
        # self.image =pygame.transform.rotate(self.image,-90)
        # self.orig_image = self.image
        if image_name==None:
            width, height=size
            self.rect=pygame.Rect(0,0,width, height)
        else:
            self.image=image_name
            self.image = pygame.transform.scale(self.image, size)
            self.image =pygame.transform.rotate(self.image,-90)
            self.orig_image = self.image
            self.rect = self.image.get_rect()
        self.speed=speed
        #死亡动画
        self.timer=0
        self.index=0
    def update(self):
        self.rect.y+=self.speed

class BackgroundSprite(GameSprite):
    def update(self):
        super().update()
        if self.rect.y>=C.SCREEN_H:
            self.rect.y=-self.rect.height

class Enemy(GameSprite):
    '''敌机'''
    def __init__(self, image=None):
        super(Enemy, self).__init__(image_name=image, size=(20, 20))
        # 自方的初始位置
        self.rect.center =random.randint(100,600),random.randint(350,500)
        self.init_x, self.init_y = self.rect.center
        self.speed_x = random.uniform(-5 / C.FPS, 5 / C.FPS)
        self.speed_y = random.uniform(15 / C.FPS, 20 / C.FPS)
        self.speed_z = 0  # random.uniform(10 / C.FPS, 15 / C.FPS)
        self.theta = random.uniform(0, 2 * math.pi)
        self.F = 0
        self.posx, self.posy = self.rect.center
        self.dead = False
        self.die_time = 0
        self.vision = 120  # 视野
        self.volume = 50  # 备弹
        self.bulleted_num = 0  # 被击中的子弹数
        self.damaged = 0  # 受到的伤害
        self.blood = 100
        self.healthy = self.blood
        # 胜利标志位
        self.win = False
        # 用于碰撞检测的变量
        self.enemies = []
        # 创建子弹 精灵组，这是个组
        self.bullets = pygame.sprite.Group()

    def update(self,action,Render=False):
        a=action[0]
        phi=action[1]
        if not self.dead:
            self.speed=self.speed+0.3*a*dt
            self.theta=self.theta+0.6*phi*dt
            self.speed = np.clip(self.speed , 10, 20)
            if self.theta>2*math.pi:
                self.theta=self.theta-2*math.pi
            elif self.theta<0:
                self.theta = self.theta + 2*math.pi
            self.posx += self.speed*math.cos(self.theta)*dt
            self.posy -= self.speed*math.sin(self.theta)*dt
            if Render:
                self.rotate()
            # self.z += self.speed_z
        if self.posx<=C.ENEMY_AREA_X:
            self.posx=C.ENEMY_AREA_X
            # self.die()
            # self.kill()
        elif self.posx>=(C.ENEMY_AREA_X+C.ENEMY_AREA_WITH):
            self.posx =C.ENEMY_AREA_X+C.ENEMY_AREA_WITH
            # self.die()
            # self.kill()
        if self.posy>=(C.ENEMY_AREA_Y+C.ENEMY_AREA_HEIGHT):
            self.posy = C.ENEMY_AREA_Y+C.ENEMY_AREA_HEIGHT
            # self.die()
            # self.kill()
        elif self.posy<=C.ENEMY_AREA_Y:
            self.posy = C.ENEMY_AREA_Y
            # self.die()
            # self.kill()
        self.rect.center = self.posx, self.posy
    def rotate(self):
        """Rotate the image of the sprite around its center."""
        # `rotozoom` usually looks nicer than `rotate`. Pygame's rotation
        # functions return new images and don't modify the originals.
        self.image = pygame.transform.rotozoom(self.orig_image, self.theta*57.3, 1)
        # Create a new rect with the center of the old rect.
        self.rect = self.image.get_rect(center=self.rect.center)

    def fire(self,speed_x,speed_y,range,volume):
        # 这里创建 子弹
        bullet1=Bullet(speed_x,speed_y,range,volume)
        #子弹的初始位置
        bullet1.rect.bottom=self.rect.centery+20
        bullet1.rect.centerx=self.rect.centerx
        #把子弹 添加到精灵组中
        self.bullets.add(bullet1)
        # print('...')
    def die(self):
        if not self.dead:
            self.die_time = pygame.time.get_ticks()
        self.dead = True




class Enemy_Group(pygame.sprite.Group):
    def __init__(self):
        super(Enemy_Group, self).__init__()


    def draw(self, surface):
        super().draw(surface)
        sprites = self.sprites()
        self.spritedict.update(zip(sprites,surface.blits((info.Info.create_label(self,label='{}'.format(spr.rect.center),size=10,color=C.RED),(spr.rect.x,spr.rect.y+spr.rect.height))for spr in sprites)))
        # surface.blit(info.Info.create_label('的',size=10,color=C.RED),)
        # print(z)


class Hero(GameSprite):
    def __init__(self,image=None):
        super(Hero, self).__init__(image_name=image,speed=random.randint(10,20),size=(20,20))
        #自方的初始位置
        self.z=1000
        self.rect.center=random.randint(300,400),random.randint(500,550)
        self.init_x,self.init_y=self.rect.center
        # self.rect.centerx=450#random.randint(C.ENEMY_AREA_X,C.SCREEN_W-C.ENEMY_AREA_X)
        # self.rect.bottom=300#C.SCREEN_H-300
        self.speed_x=random.uniform(-5/C.FPS,5/C.FPS)
        self.speed_y= random.uniform(15/C.FPS, 20/C.FPS)
        self.speed_z = 0#random.uniform(10 / C.FPS, 15 / C.FPS)
        self.theta=random.uniform(0,2*math.pi)
        self.F = 0
        self.posx, self.posy = self.rect.center
        self.dead=False
        self.die_time = 0
        self.vision = 120  # 视野
        self.volume = 50  # 备弹
        self.bulleted_num=0#被击中的子弹数
        self.damaged = 0  # 受到的伤害
        self.blood=100
        self.healthy=self.blood
        #胜利标志位
        self.win=False
        #用于碰撞检测的变量
        self.enemies=[]
        #创建子弹 精灵组，这是个组
        self.bullets=pygame.sprite.Group()

    def update(self,action,Render=False):
        a=action[0]
        phi=action[1]
        if not self.dead:
            self.speed=self.speed+0.6*a*dt
            self.theta=self.theta+1.2*phi*dt
            self.speed = np.clip(self.speed , 10, 20)
            if self.theta>2*math.pi:
                self.theta=self.theta-2*math.pi
            elif self.theta<0:
                self.theta = self.theta + 2*math.pi
            self.posx += self.speed*math.cos(self.theta)*dt
            self.posy -= self.speed*math.sin(self.theta)*dt
            if Render:
                self.rotate()
            # self.z += self.speed_z
        if self.posx<=C.ENEMY_AREA_X:
            self.posx=C.ENEMY_AREA_X
            # self.die()
            # self.kill()
        elif self.posx>=(C.ENEMY_AREA_X+C.ENEMY_AREA_WITH):
            self.posx =C.ENEMY_AREA_X+C.ENEMY_AREA_WITH
            # self.die()
            # self.kill()
        if self.posy>=(C.ENEMY_AREA_Y+C.ENEMY_AREA_HEIGHT):
            self.posy = C.ENEMY_AREA_Y+C.ENEMY_AREA_HEIGHT
            # self.die()
            # self.kill()
        elif self.posy<=C.ENEMY_AREA_Y:
            self.posy = C.ENEMY_AREA_Y
            # self.die()
            # self.kill()
        self.rect.center = self.posx, self.posy
    def rotate(self):
        """Rotate the image of the sprite around its center."""
        # `rotozoom` usually looks nicer than `rotate`. Pygame's rotation
        # functions return new images and don't modify the originals.
        self.image = pygame.transform.rotozoom(self.orig_image, self.theta*57.3, 1)
        # Create a new rect with the center of the old rect.
        self.rect = self.image.get_rect(center=self.rect.center)

    def fire(self,speed_x,speed_y,range,volume):
        # 这里创建 子弹
        bullet1=Bullet(speed_x,speed_y,range,volume)
        #子弹的初始位置
        bullet1.rect.bottom=self.rect.centery-20
        bullet1.rect.centerx=self.rect.centerx
        #把子弹 添加到精灵组中
        self.bullets.add(bullet1)
        # print('...')

    def die(self):
        if not self.dead:
            self.die_time = pygame.time.get_ticks()
        self.dead = True
        # self.image=set_up.GRAPHICS['hero_blowup_n4']
        # self.image = pygame.transform.scale(self.image, size=(30,30))
        # print('死')

    # def die_image(self):
    #     image1 = set_up.GRAPHICS['hero_blowup_n1']
    #     image1 = pygame.transform.scale(image1, size=(30, 30))
    #     image2 = set_up.GRAPHICS['hero_blowup_n2']
    #     image2 = pygame.transform.scale(image2, size=(30, 30))
    #     image3 = set_up.GRAPHICS['hero_blowup_n3']
    #     image3 = pygame.transform.scale(image3, size=(30, 30))
    #     image4 = set_up.GRAPHICS['hero_blowup_n4']
    #     image4 = pygame.transform.scale(image4, size=(30, 30))
    #     image_list=[image1,image2,image3,image4]
    #     time_pause = [125, 125, 125, 125]  # 帧与帧之间的停留时间，即第一幅图停留500ms，第二幅图停留500ms
    #     current_time = pygame.time.get_ticks()  # 到得当前时间
    #     if self.timer==0:
    #         self.timer=current_time
    #     elif current_time - self.timer > time_pause[self.index]:  # 当前与上一刻记录的时间之差大于500ms，index就加1
    #         self.index += 1
    #         self.index %= 4
    #         self.timer = current_time
    #     self.image=image_list[self.index]

    # def __del__(self):
    #     print('ji %s'%self.rect.x)
class Obstacle(GameSprite):
    '''障碍'''
    def __init__(self,image=None):
        super(Obstacle, self).__init__(image_name=image,size=(40,40))
        #自方的初始位置
        self.rect.center=random.randint(100,600),random.randint(300,500)
        self.init_x,self.init_y=self.rect.center

class Goal(GameSprite):
    '''障碍'''
    def __init__(self,image=None):
        super(Goal, self).__init__(image_name=image,size=(20,20))
        #自方的初始位置
        self.rect.center=random.randint(150,500),random.randint(150,300)
        self.init_x,self.init_y=self.rect.center

class Bullet(GameSprite):
    '''子弹'''
    def __init__(self,speed_x=0,speed_y=-C.BULLET_SPEED,range=30,volume=500,rate=600,damage=50):
        #一架飞机子弹数量为500，一秒钟射10发，射程1200m，一颗子弹伤害为50
        super(Bullet, self).__init__(image_name='bullet1',size=(4,4)) #航空机炮的子弹速度约为900m/s
        self.speed_x = speed_x
        self.speed_y = speed_y
        self.range=range
        self.volume=volume
        self.rate=rate
        self.damage=damage
        self.dis_x,self.dis_y=0,0


    def update(self):
        #子弹位置更新
        self.rect.x += self.speed_x
        self.rect.y+=self.speed_y
        #计算子弹飞行距离
        self.dis_x+=self.speed_x
        self.dis_y+=self.speed_y
        dis=math.hypot(self.dis_x,self.dis_y)
        # print(self.dis_x,self.dis_y,dis)
        #子弹飞出屏幕
        if dis>self.range:
            self.kill()
        # if self.rect.bottom<0:
        #     self.kill()
    
    # def __del__(self):
    #     print('子弹寄')
