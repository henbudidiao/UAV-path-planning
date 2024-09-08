# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:23
import sys

import numpy as np
from numpy import deg2rad

import matplotlib.pyplot as plt
import os
sys.path.append('E:\githubwork\AeroBenchVVPython-master\code')
# os.chdir('E:\githubwork\AeroBenchVVPython-master\code')
from aerobench.run_f16_sim import run_f16_sim
from aerobench.visualize import anim3d, plot
from aerobench.examples.waypoint.waypoint_autopilot import WaypointAutopilot

FPS = 60


def simulate(filename):
    'simulate the system, returning waypoints, res'

    ### Initial Conditions ###
    power = 9  # engine power level (0-10)

    # Default alpha & beta
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 1500  # altitude (ft)
    vt = 540  # initial velocity (ft/sec)
    phi = 0  # Roll angle from wings level (rad)
    theta = 0  # Pitch angle from nose level (rad)
    psi = 0  # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [vt, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    init = [vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power]
    tmax = 300  # simulation time

    # make waypoint list
    waypoints = [[-5000, -10000, alt],
                 [-10000, -7500, alt - 500],
                 [-15000, 3000, alt - 200],
                 [0, 8000, alt + 1000]]

    ap = WaypointAutopilot(waypoints, stdout=True)

    step = 1 / FPS
    extended_states = True
    res = run_f16_sim(init, tmax, ap, step=step, extended_states=extended_states, integrator_str='rk45')

    # res['states'][:,0],res['states'][:,9],res['states'][:,10],res['states'][:,11]=res['states'][:,0]*0.3048,res['states'][:,9]*0.3048,res['states'][:,10]*0.3048,res['states'][:,11]*0.3048

    print(f"Waypoint simulation completed in {round(res['runtime'], 2)} seconds (extended_states={extended_states})")

    if filename.endswith('.mp4'):
        skip_override = 4
    elif filename.endswith('.gif'):
        skip_override = 15
    else:
        skip_override = 10

    anim_lines = []
    modes = res['modes']
    modes = modes[0::skip_override]

    def init_extra(ax):
        'initialize plot extra shapes'

        l1, = ax.plot([], [], [], 'bo', ms=8, lw=0, zorder=50)
        anim_lines.append(l1)

        l2, = ax.plot([], [], [], 'lime', marker='o', ms=8, lw=0, zorder=50)
        anim_lines.append(l2)

        return anim_lines

    def update_extra(frame):
        'update plot extra shapes'

        mode_names = ['Waypoint 1', 'Waypoint 2', 'Waypoint 3', 'Waypoint 4']

        done_xs = []
        done_ys = []
        done_zs = []

        blue_xs = []
        blue_ys = []
        blue_zs = []

        for i, mode_name in enumerate(mode_names):
            if modes[frame] == mode_name:
                blue_xs.append(waypoints[i][0])
                blue_ys.append(waypoints[i][1])
                blue_zs.append(waypoints[i][2])
                break

            done_xs.append(waypoints[i][0])
            done_ys.append(waypoints[i][1])
            done_zs.append(waypoints[i][2])

        anim_lines[0].set_data(blue_xs, blue_ys)
        anim_lines[0].set_3d_properties(blue_zs)

        anim_lines[1].set_data(done_xs, done_ys)
        anim_lines[1].set_3d_properties(done_zs)

    return res, init_extra, update_extra, skip_override, waypoints


def main():
    'main function'

    if len(sys.argv) > 1 and (sys.argv[1].endswith('.mp4') or sys.argv[1].endswith('.gif')):
        filename = sys.argv[1]
        print(f"saving result to '{filename}'")
    else:
        filename = ''
        print("Plotting to the screen. To save a video, pass a command-line argument ending with '.mp4' or '.gif'.")

    res, init_extra, update_extra, skip_override, waypoints = simulate(filename)

    # # 画两张图
    # plot.plot_single(res, 'alt', title='Altitude (ft)')
    # alt_filename = 'waypoint_altitude.png'
    # plt.savefig(alt_filename)
    # print(f"Made {alt_filename}")
    # plt.close()
    #
    # plot.plot_overhead(res, waypoints=waypoints)
    # overhead_filename = 'waypoint_overhead.png'
    # plt.savefig(overhead_filename)
    # print(f"Made {overhead_filename}")
    # plt.close()

    anim3d.make_anim(res, filename, f16_scale=70, viewsize=5000, viewsize_z=4000, trail_pts=np.inf,
                     elev=27, azim=-107, skip_frames=skip_override,
                     chase=True, fixed_floor=True, init_extra=init_extra, update_extra=update_extra)


if __name__ == '__main__':
    main()
# import pygame as pg
#
#
# class Entity(pg.sprite.Sprite):
#
#     def __init__(self, pos):
#         super().__init__()
#         self.image = pg.Surface((122, 70), pg.SRCALPHA)
#         pg.draw.polygon(self.image, pg.Color('dodgerblue1'),
#                         ((1, 0), (120, 35), (1, 70)))
#         # A reference to the original image to preserve the quality.
#         self.orig_image = self.image
#         self.rect = self.image.get_rect(center=pos)
#         self.angle = 0
#
#     def update(self):
#         self.angle += 2
#         self.rotate()
#
#     def rotate(self):
#         """Rotate the image of the sprite around its center."""
#         # `rotozoom` usually looks nicer than `rotate`. Pygame's rotation
#         # functions return new images and don't modify the originals.
#         self.image = pg.transform.rotozoom(self.orig_image, self.angle, 1)
#         # Create a new rect with the center of the old rect.
#         self.rect = self.image.get_rect(center=self.rect.center)
#
# def main():
#     screen = pg.display.set_mode((640, 480))
#     clock = pg.time.Clock()
#     all_sprites = pg.sprite.Group(Entity((320, 240)))
#
#     while True:
#         for event in pg.event.get():
#             if event.type == pg.QUIT:
#                 return
#
#         all_sprites.update()
#         screen.fill((30, 30, 30))
#         all_sprites.draw(screen)
#         pg.display.flip()
#         clock.tick(30)
#
#
# if __name__ == '__main__':
#     pg.init()
#     main()
#     pg.quit()