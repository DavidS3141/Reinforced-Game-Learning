#!/usr/bin/env python
'''main.py: Main script for training session.'''
###############################################################################

import sys
import select
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from games.ultimate_tictactoe.ultimate_tictactoe import (
    Ultimate_Game as Game,
)
from games.ultimate_tictactoe.ultimate_tictactoe_ai import (
    random_ai,
    semi_random_ai,
)
from model.inference import inference
from model.loss import loss
from model.training import training
from copy import deepcopy as copy


def smooth_data(x, b, factor=1):
    res = []
    c_val = 0.
    for i, val in enumerate(x):
        c_val = c_val * b + val * (1. - b)
        res.append(c_val * factor / (1 - b**(i + 1)))
    return res


def get_action_id_from_values(values, riskyness=0.01):
    v1 = np.max(values)
    idx1 = np.argmax(values)
    idx2 = np.random.randint(len(values))
    v2 = values[idx2]
    if idx1 == idx2:
        return idx1
    if v1 - v2 <= riskyness:
        select = np.random.randint(2)
    else:
        tresh = v1 - riskyness
        p1 = (tresh - v2) / (v1 - v2)
        if p1 < 0.5:
            p1 = 0.5
        select = 1
        if p1 >= np.random.rand():
            select = 0
    if select == 0:
        return idx1
    return idx2


def get_nn_action(game):
    player_turn = game.get_player_turn()
    action_list = game.get_action_list()
    states = []
    for act in action_list:
        co = copy(game)
        co.take_action(player_turn, act)
        state = co.get_state_for_player(player_turn)
        states.append(state)
    val = sess.run(net_out, feed_dict={net_in: states})
    idx = get_action_id_from_values(val)
    return action_list[idx]


def play_game(p1, p2):
    game = Game()
    status = game.get_status()
    while status == -1:
        if game.get_player_turn() == 0:
            a = p1(game)
        else:
            a = p2(game)
        game.take_action(game.get_player_turn(), a)
        status = game.get_status()
    res = status
    if status == -2:
        res = 0.5
    elif status == -3:
        res = game.get_player_turn()
    return res


state_dim = Game.state_dim
learning_rate = 1e-3

net_in = tf.placeholder(tf.float32, shape=(None, state_dim))
net_labels = tf.placeholder(tf.float32, shape=(None, 1))
net_out, net_logits = inference(net_in, state_dim)
net_loss = loss(net_logits, net_labels)
net_trainer = training(net_loss, learning_rate)
rounds_played = []
loss_list = []
random_eval = []
semi_random_eval = []
ai_eval = []
eval_counter = []

with tf.Session() as sess:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('data/', sess.graph)
    sess.run(tf.global_variables_initializer())
    for train_step in range(100000):
        if train_step % 10000 == 0:
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
        else:
            human = False
        game = Game()
        playersTurn = 0
        status = -1
        game_states = [[], [game.get_state_for_player(1)]]
        round_count = 0
        while status == -1:
            round_count += 1
            if human and playersTurn == 1:
                game.visualize()
                xs, ys, x, y = input().split(' ')
                a = 27 * int(xs) + 9 * int(ys) + 3 * int(x) + int(y)
            else:
                a = get_nn_action(game)
            game.take_action(playersTurn, a)
            status = game.get_status()
            if status == -3:
                status = 1 - playersTurn
            game_states[playersTurn].append(
                game.get_state_for_player(playersTurn))
            playersTurn = 1 - playersTurn
        rounds_played.append(round_count)
        n0 = len(game_states[0])
        n1 = len(game_states[1])
        l0 = np.ones((n0))
        l1 = np.ones((n1))
        if status == -2:
            l0 *= 0.5
            l1 *= 0.5
        elif status == 1:
            l0 *= 0
            l1 *= 1
        elif status == 0:
            l0 *= 1
            l1 *= 0
        else:
            raise Exception('Wrong end game status!')
        game_states = game_states[0] + game_states[1]
        labels = np.concatenate([l0, l1]).reshape([n0 + n1, 1])
        _, preds, losses = sess.run(
            [net_trainer, net_out, net_loss],
            feed_dict={net_in: game_states, net_labels: labels})
        loss_list.append(losses)
        if train_step % 100 == 0:
            eval_counter.append(train_step)
            # ai_eval.append(play_game(ai,get_nn_action)+1-play_game(get_nn_action,ai))
            random_eval.append(play_game(random_ai, get_nn_action) +
                               1 - play_game(get_nn_action, random_ai))
            semi_random_eval.append(play_game(
                semi_random_ai, get_nn_action) + 1
                - play_game(get_nn_action, semi_random_ai))
        if train_step % 100 == 0:
            saver.save(sess, '../data/ultimate_tictactoe/ut',
                       global_step=train_step)
            plt.plot(smooth_data(rounds_played, .99),
                     'b', label='rounds played')
            plt.plot(smooth_data(rounds_played, .999), 'b')
            plt.plot(smooth_data(loss_list, .99), 'c', label='losses')
            plt.plot(smooth_data(loss_list, .999), 'c')
            plt.plot(eval_counter, smooth_data(semi_random_eval, .9,
                                               factor=30), 'r',
                     label='semi-random eval')
            plt.plot(eval_counter, smooth_data(
                semi_random_eval, .99, factor=30), 'r')
            plt.plot(eval_counter, smooth_data(
                random_eval, .9, factor=30), 'm', label='random eval')
            plt.plot(eval_counter, smooth_data(
                random_eval, .99, factor=30), 'm')
            plt.legend(loc='lower left')
            plt.grid()
            plt.savefig('../data/ultimate_tictactoe/evolution.png')
            plt.close()
            game.visualize()
            print(losses)
