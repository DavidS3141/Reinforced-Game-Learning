#!/user/bin/env python

'''tictactoe_ai.py: Implement an ai for the game of Tic-Tac-Toe.'''

################################################################################

from copy import deepcopy as copy
from numpy import random
import numpy as np

def random_ai(game):
    return random.choice(game.get_action_list())

def semi_random_ai(game):
    alist = game.get_action_list()
    p_turn = game.get_player_turn()
    prev = np.sum(game.superboard[p_turn,:,:])
    good_actions = []
    for a in alist:
        co = copy(game)
        co.take_action(p_turn, a)
        stat = co.get_status()
        if stat == p_turn:
            return a
        now = np.sum(co.superboard[p_turn,:,:])
        assert(now>=prev)
        assert(now<=prev+1)
        if now == prev+1:
            good_actions.append(a)
    if len(good_actions)>0:
        return random.choice(good_actions)
    else:
        return random.choice(alist)

if __name__ == '__main__':
    from ultimate_tictactoe import Ultimate_TicTacToe

    g = Ultimate_TicTacToe()
    turn = 0
    status = g.get_status()
    print('Do you want to start? [Y/n]')
    player_start = True
    answer = input()
    if answer == 'n' or answer == 'N':
        player_start = False
    elif answer == 'y' or answer == 'Y' or len(answer)==0:
        player_start = True
    else:
        print('Invalid answer! Exit!')
        quit()
    player_id = 0
    if not player_start:
        player_id = 1

    while status==-1:
        g.visualize()
        if turn == player_id:
            # x,y = input().split(' ')
            # act = 3*int(x)+int(y)
            # act = random_ai(g)
            act = semi_random_ai(g)
        else:
            act = random_ai(g)
        g.take_action(turn, act)
        status = g.get_status()
        turn = 1-turn

    g.visualize()
    print(status)
