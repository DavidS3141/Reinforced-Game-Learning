#!/user/bin/env python

'''main.py: Main script for training session.'''

################################################################################

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from games.tictactoe.tictactoe import TicTacToe as Game
from model.inference import inference
from model.loss import loss
from model.training import training
from copy import deepcopy as copy

def get_action_id_from_values(values, riskyness = 0.01):
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
        p1 = (tresh - v2)/(v1-v2)
        if p1 < 0.5:
            p1 = 0.5
        select = 1
        if p1>= np.random.rand():
            select = 0
    if select == 0:
        return idx1
    return idx2

state_dim = Game.state_dim
learning_rate = 1e-3

net_in = tf.placeholder(tf.float32, shape=(None,state_dim))
net_labels = tf.placeholder(tf.float32, shape=(None,1))
net_out, net_logits = inference(net_in)
net_loss = loss(net_logits, net_labels)
net_trainer = training(net_loss, learning_rate)
rounds_played = []
loss_list = []

with tf.Session() as sess:

    train_writer = tf.summary.FileWriter('data/',
                                      sess.graph)

    sess.run(tf.global_variables_initializer())
    for train_step in range(1000000):
        if train_step % 10000 == 0:
            human = False
        else:
            human = False
        game = Game()
        playersTurn = 0
        status = -1
        game_states = [[],[game.get_state_for_player(1)]]
        round_count = 0
        while status == -1:
            round_count += 1
            if human and playersTurn == 1:
                game.visualize()
                a = int(input())
            else:
                action_list = game.get_action_list()
                states = []
                for act in action_list:
                    co = copy(game)
                    co.take_action(playersTurn, act)
                    state = co.get_state_for_player(playersTurn)
                    states.append(state)
                val = sess.run(net_out, feed_dict={net_in: states})
                idx = get_action_id_from_values(val)
                a = action_list[idx]
            status = game.take_action(playersTurn, a)
            if status == -3:
                status = 1-playersTurn
            game_states[playersTurn].append(game.get_state_for_player(playersTurn))
            playersTurn = 1-playersTurn
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
        game_states = game_states[0]+game_states[1]
        labels = np.concatenate([l0, l1]).reshape([n0+n1,1])
        _, preds, losses = sess.run([net_trainer, net_out, net_loss], feed_dict={net_in: game_states, net_labels:labels})
        loss_list.append(losses)
        if train_step%100==0:
            plt.plot(rounds_played, label='rounds played')
            plt.plot(loss_list, label='losses')
            plt.legend()
            plt.savefig('evolution.png')
            plt.close()
            game.visualize()
            print(game_states)
            print(losses)
