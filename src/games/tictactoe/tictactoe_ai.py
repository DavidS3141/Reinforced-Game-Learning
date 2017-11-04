#!/user/bin/env python

'''tictactoe_ai.py: Implement an ai for the game of Tic-Tac-Toe.'''

################################################################################

from copy import deepcopy as copy
from numpy import random

def get_possible_actions(game):
    state = game.get_state_for_player(0)
    state = state.reshape([3,3,3])
    poss_actions = []
    for a in range(9):
        x,y = (a//3, a%3)
        if state[2,x,y] == 1:
            poss_actions.append(a)
    return poss_actions

def eval_pos(game):
    status = game.get_status()
    if status != -1:
        if status == -2:
            return 0.5, [-1]
        elif status == -3:
            return game.get_player_turn(), [-1]
        return status, [-1]
    actions = get_possible_actions(game)
    if len(actions)==9:
        return 0.5, actions
    if len(actions)==8:
        state = game.get_state_for_player(0)
        state = state.reshape([3,3,3])
        if state[0,1,1]==1:
            return 0.5, [0,2,6,8]
        elif state[0,0,0]==1 or state[0,0,2]==1 or state[0,2,0]==1 or state[0,2,2]==1:
            return 0.5, [4]
        elif state[0,0,1]==1:
            return 0.5, [0,2,4,7]
        elif state[0,1,0]==1:
            return 0.5, [0,4,5,6]
        elif state[0,1,2]==1:
            return 0.5, [2,3,4,8]
        elif state[0,2,1]==1:
            return 0.5, [1,4,6,8]
        else:
            raise Exception('Logic error!')
    turn = game.get_player_turn()
    result = 100.
    best_actions = [-1]
    for a in actions:
        cop = copy(game)
        cop.take_action(turn, a)
        local_result, _ = eval_pos(cop)
        if abs(local_result-turn)<abs(result-turn):
            result = local_result
            best_actions = [a]
        elif abs(local_result-turn)==abs(result-turn):
            assert(local_result == result)
            best_actions.append(a)
    return result, best_actions

def ai(game):
    _, best_actions = eval_pos(game)
    return random.choice(best_actions)

def random_ai(game):
    return random.choice(game.get_action_list())

def semi_random_ai(game):
    return random.choice(get_possible_actions(game))

if __name__ == '__main__':
    from tictactoe import TicTacToe

    g = TicTacToe()
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
            act = semi_random_ai(g)
        else:
            act = ai(g)
        g.take_action(turn, act)
        status = g.get_status()
        turn = 1-turn

    g.visualize()
