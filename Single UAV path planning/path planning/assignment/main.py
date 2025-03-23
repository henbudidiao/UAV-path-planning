# -*- coding: utf-8 -*-
#开发者：Bright Fang
#开发时间：2022/9/17 18:21
from assignment import tools,set_up
from assignment.states import main_menu,load_screen,battle_screen

def main():
    state_dict={
        # 'main_menu':main_menu.MainMenu(),
        #         'load_screen':load_screen.LoadScreen(),
                'battle_screen':battle_screen.BattleScreen(n=1,m=0),
                'game_over':load_screen.GameOver()
                }
    game=tools.Game(state_dict,'battle_screen')

    game.run()

if __name__ == '__main__':
    main()