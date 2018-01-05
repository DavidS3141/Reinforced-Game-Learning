#!/usr/bin/env python
'''
main-zero.py: Main script for training session similar to AlphaGo Zero.
'''
###############################################################################
from algorithm.mcts_network import MCTS_Network
from algorithm.mcts_player import MCTS_Player

from tqdm import tqdm
import sys
import select
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from games.ultimate_tictactoe.ultimate_tictactoe import Game
from games.ultimate_tictactoe.ultimate_tictactoe_ai import (
    Random_AI,
    Semi_Random_AI,
)


def smooth_data(x, b, factor=1):
    res = []
    c_val = 0.
    for i, val in enumerate(x):
        c_val = c_val * b + val * (1. - b)
        res.append(c_val * factor / (1 - b**(i + 1)))
    return res


def play_game(players):
    '''
    players is a list of player objects having the following methods:
    -set_player_id(player_id)
    -get_action(game)
    -set_next_action(game, action)
    -set_result(result)
    '''
    for player_id, player in enumerate(players):
        player.set_player_id(player_id)
    game = Game()
    taken_actions = 0
    while not game.is_terminal():
        turn = game.get_player_turn()
        action = players[turn].get_action(game)
        for player in players:
            player.set_next_action(game, action)
        game.take_action(turn, action)
        taken_actions += 1
    result = game.get_points()
    for player in players:
        player.set_result(result)
    return result, taken_actions


# state_dim = Game.state_dim
# learning_rate = 1e-3
#
# net_in = tf.placeholder(tf.float32, shape=(None, state_dim))
# net_labels = tf.placeholder(tf.float32, shape=(None, 1))
# net_out, net_logits = inference(net_in, state_dim)
# net_loss = loss(net_logits, net_labels)
# net_trainer = training(net_loss, learning_rate)
# rounds_played = []
# loss_list = []
# random_eval = []
# semi_random_eval = []
# ai_eval = []
# eval_counter = []
#
# with tf.Session() as sess:
#     saver = tf.train.Saver()
#     train_writer = tf.summary.FileWriter('data/', sess.graph)
#     sess.run(tf.global_variables_initializer())
#     for train_step in range(100000):
#         if train_step % 10000 == 0:
#             print('Do you want to play? [N/y] (10 sec)')
#             i, o, e = select.select([sys.stdin], [], [], 10)
#             if (i):
#                 res = sys.stdin.readline().strip()
#             else:
#                 res = 'N'
#             if res == 'N' or res == 'n' or len(res) == 0:
#                 human = False
#             elif res == 'y' or res == 'Y':
#                 human = True
#             else:
#                 print('Wrong answer! Exit!')
#                 quit()
#         else:
#             human = False
#         game = Game()
#         playersTurn = 0
#         status = -1
#         game_states = [[], [game.get_state_for_player(1)]]
#         round_count = 0
#         while status == -1:
#             round_count += 1
#             if human and playersTurn == 1:
#                 game.visualize()
#                 xs, ys, x, y = input().split(' ')
#                 a = 27 * int(xs) + 9 * int(ys) + 3 * int(x) + int(y)
#             else:
#                 a = get_nn_action(game)
#             game.take_action(playersTurn, a)
#             status = game.get_status()
#             if status == -3:
#                 status = 1 - playersTurn
#             game_states[playersTurn].append(
#                 game.get_state_for_player(playersTurn))
#             playersTurn = 1 - playersTurn
#         rounds_played.append(round_count)
#         n0 = len(game_states[0])
#         n1 = len(game_states[1])
#         l0 = np.ones((n0))
#         l1 = np.ones((n1))
#         if status == -2:
#             l0 *= 0.5
#             l1 *= 0.5
#         elif status == 1:
#             l0 *= 0
#             l1 *= 1
#         elif status == 0:
#             l0 *= 1
#             l1 *= 0
#         else:
#             raise Exception('Wrong end game status!')
#         game_states = game_states[0] + game_states[1]
#         labels = np.concatenate([l0, l1]).reshape([n0 + n1, 1])
#         _, preds, losses = sess.run(
#             [net_trainer, net_out, net_loss],
#             feed_dict={net_in: game_states, net_labels: labels})
#         loss_list.append(losses)
#         if train_step % 100 == 0:
#             eval_counter.append(train_step)
#             ai_eval.append(
#                 play_game(ai,get_nn_action)+1-play_game(get_nn_action,ai))
#             random_eval.append(play_game(random_ai, get_nn_action) +
#                                1 - play_game(get_nn_action, random_ai))
#             semi_random_eval.append(play_game(
#                 semi_random_ai, get_nn_action) + 1
#                 - play_game(get_nn_action, semi_random_ai))
#         if train_step % 100 == 0:
#             saver.save(sess, '../data/ultimate_tictactoe/ut',
#                        global_step=train_step)
#             plt.plot(smooth_data(rounds_played, .99),
#                      'b', label='rounds played')
#             plt.plot(smooth_data(rounds_played, .999), 'b')
#             plt.plot(smooth_data(loss_list, .99), 'c', label='losses')
#             plt.plot(smooth_data(loss_list, .999), 'c')
#             plt.plot(eval_counter, smooth_data(semi_random_eval, .9,
#                                                factor=30), 'r',
#                      label='semi-random eval')
#             plt.plot(eval_counter, smooth_data(
#                 semi_random_eval, .99, factor=30), 'r')
#             plt.plot(eval_counter, smooth_data(
#                 random_eval, .9, factor=30), 'm', label='random eval')
#             plt.plot(eval_counter, smooth_data(
#                 random_eval, .99, factor=30), 'm')
#             plt.legend(loc='lower left')
#             plt.grid()
#             plt.savefig('../data/ultimate_tictactoe/evolution.png')
#             plt.close()
#             game.visualize()
#             print(losses)


def do_selfplay(mcts_net):
    players = []
    for i in range(Game.nbr_players):
        players.append(MCTS_Player(mcts_net))
    play_game(players)
    in_train = []
    priors = []
    values = []
    for player in players:
        in_, priors_, values_ = player.extract_train_data()
        in_train += in_
        priors += priors_
        values += values_
    return in_train, priors, values


class Human_Player(object):
    def set_player_id(self, player_id):
        self.player_id = player_id
        print('You are player %d!' % player_id)

    def get_action(self, game):
        game.visualize()
        print('Where do you want to set?')
        return game.user_input_2_action()

    def set_next_action(self, game, action):
        print('Player played action %s!' % game.action_2_user_output(action))

    def set_result(self, result):
        print('You got %.1f points!' % result[self.player_id])


if __name__ == '__main__':
    max_epochs = 2**12
    selfplays_per_epoch = 1
    max_selfplay_history = 2**9
    batch_size = 2**5
    network = MCTS_Network(in_dim=Game.state_dim,
                           priors_dim=Game.max_nbr_actions,
                           values_dim=Game.nbr_players,
                           batch_size=batch_size)
    in_train = []
    out_train_priors = []
    out_train_values = []
    eval_counters = []
    random_eval = []
    semi_random_eval = []
    lengths = []
    for epoch in tqdm(range(max_epochs)):
        print('Doing selfplays ...')
        for i in range(selfplays_per_epoch):
            in_, priors_, values_ = do_selfplay(network)
            in_train += in_
            out_train_priors += priors_
            out_train_values += values_
        print('Finished selfplays!')
        if len(in_train) > max_selfplay_history:
            in_train = in_train[-max_selfplay_history:]
            out_train_priors = out_train_priors[-max_selfplay_history:]
            out_train_values = out_train_values[-max_selfplay_history:]
        print('length of selfplay history training samples: %d' %
              len(in_train))
        print('Train network ...')
        network.train(in_train, out_train_priors, out_train_values)
        print('Finished training!')
        if epoch % 1 == 0:
            print('Eval network ...')
            eval_counters.append(epoch)
            result, length = play_game([MCTS_Player(network, explore=False),
                                        Random_AI()])
            random_eval.append(result[0])
            result, length2 = play_game([MCTS_Player(network, explore=False),
                                        Semi_Random_AI()])
            semi_random_eval.append(result[0])
            lengths.append((length + length2) * 0.01)
            print('Finished evaluation!')
        if epoch % 10 == 0:
            p = plt.plot(eval_counters, smooth_data(semi_random_eval, .99),
                         label='semi-random eval')
            plt.plot(eval_counters, smooth_data(semi_random_eval, .999),
                     color=p[0].get_color())
            p = plt.plot(eval_counters, smooth_data(random_eval, .99),
                         label='random eval')
            plt.plot(eval_counters, smooth_data(random_eval, .999),
                     color=p[0].get_color())
            p = plt.plot(eval_counters, smooth_data(lengths, .99),
                         label='lengths')
            plt.plot(eval_counters, smooth_data(lengths, .999),
                     color=p[0].get_color())
            plt.legend(loc='lower left')
            plt.grid()
            plt.savefig('../data_mcts/ultimate_tictactoe/evolution.png')
            plt.close()
        if epoch % 100 == 0:
            print('Do you want to play? [N/y] (10 sec)')
            i, o, e = select.select([sys.stdin], [], [], 10)
            if (i):
                res = sys.stdin.readline().strip()
            else:
                res = 'N'
            if res == 'N' or res == 'n' or len(res) == 0:
                human = False
            elif res == 'y' or res == 'Y':
                human = True
            else:
                print('Wrong answer! Exit!')
                quit()
            if human:
                play_game([MCTS_Player(network, explore=False), Human_Player()])
    network.close()
