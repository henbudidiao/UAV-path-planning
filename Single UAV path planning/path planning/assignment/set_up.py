# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:22
import pygame
from assignment import  constants as C
from assignment import tools
pygame.init()
pygame.mixer.init()
SCREEN=pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

GRAPHICS=tools.load_graphics('G:\path planning/assignment\source\image')

SOUND=tools.load_sound('G:\path planning/assignment\source\music')