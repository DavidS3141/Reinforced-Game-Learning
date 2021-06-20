#!/usr/bin/env python3

import numpy as np
import os
import os.path as osp
from games.game import Game, GameEngine
from players.basic_players import (  # noqa: F401
    Random,
    Human,
    MCTS_RandomPolicy,
    SuperRandom,
    MCTS_NetworkPolicy,
    RandomHuman,
)
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.layers as layers


class TicTacToe(Game):
    CIRCLE = 0
    CROSS = 1
    lines = [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1)],
        [(2, 0), (2, 1), (2, 2)],
        [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    nbr_players = 2
    state_dimensionality = 27
    max_nbr_actions = 9

    def generate_initial_state(self):
        board = np.zeros(shape=(3, 3, 3))
        board[2, :, :] = 1
        return dict(board=board, turn=self.CIRCLE)

    def flatten_state(self, state):
        if state["turn"] == 0:
            return state["board"].flatten()
        elif state["turn"] == 1:
            return state["board"][[1, 0, 2]].flatten()
        else:
            raise Exception("The player %d does not exist!" % state["turn"])

    def visualize(self, state):
        ords = ord(" ") * state["board"][2, :, :]
        ords += ord("O") * state["board"][0, :, :]
        ords += ord("X") * state["board"][1, :, :]
        ords = np.array(ords, dtype=int)
        ords[
            np.logical_and(
                np.logical_and(ords != ord(" "), ords != ord("X")), ords != ord("O")
            )
        ] = ord("#")
        lines = []
        lines.append(" %s | %s | %s " % tuple([chr(s) for s in ords[0]]))
        lines.append("---+---+---")
        lines.append(" %s | %s | %s " % tuple([chr(s) for s in ords[1]]))
        lines.append("---+---+---")
        lines.append(" %s | %s | %s " % tuple([chr(s) for s in ords[2]]))
        return "\n".join(lines)

    # def minimal_visualize(self, state):
    #     ords = ord(" ") * state["board"][2, :, :]
    #     ords += ord("O") * state["board"][0, :, :]
    #     ords += ord("X") * state["board"][1, :, :]
    #     ords = np.array(ords, dtype=int)
    #     ords[
    #         np.logical_and(
    #             np.logical_and(ords != ord(" "), ords != ord("X")), ords != ord("O")
    #         )
    #     ] = ord("#")
    #     lines = []
    #     lines.append("%s%s%s" % tuple([chr(s) for s in ords[0]]))
    #     lines.append("%s%s%s" % tuple([chr(s) for s in ords[1]]))
    #     lines.append("%s%s%s" % tuple([chr(s) for s in ords[2]]))
    #     return "\n".join(lines)

    def get_player_turn(self, state):
        return state["turn"]

    def take_action(self, state, player_id, action_id):
        if self.get_player_turn(state) != player_id:
            raise Exception("It was not player_id %d's turn!" % player_id)

        old_status = self.get_status(state)

        if old_status >= 0:
            raise Exception("Game was already won by player %d!" % old_status)
        if old_status == -2:
            raise Exception("Game was already finished with a draw!")
        if old_status == -3:
            raise Exception("Game was already finished by an invalid move!")

        x, y = self.action_id_2_xy(action_id)

        board = state["board"].copy()
        board[2, x, y] -= 1
        board[player_id, x, y] += 1
        return dict(board=board, turn=1 - state["turn"])

    def is_terminal(self, state):
        return self.get_status(state) != -1

    def get_points(self, state):
        status = self.get_status(state)
        assert status != -1
        if status == -3:  # invalid move, set status to winner id
            status = state["turn"]
        if status == 0:
            return [1.0, 0.0]
        if status == 1:
            return [0.0, 1.0]
        if status == -2:
            return [0.5, 0.5]
        raise Exception("LogicError")

    def get_status(self, state):
        assert np.max(state["board"][2, :, :]) <= 1
        assert np.min(state["board"][:2, :, :]) >= 0
        assert np.all(np.sum(state["board"], axis=0) == np.ones([3, 3]))

        if np.min(state["board"][2, :, :]) < 0:  # invalid move
            return -3

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= state["board"][self.CIRCLE][cell]
                cros_f *= state["board"][self.CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert cross_lines * circle_lines == 0

        if cross_lines == 0 and circle_lines == 0:
            if (
                np.max(state["board"][2, :, :]) == 0
            ):  # no possible move left, it is a draw
                return -2
            else:  # game is still ongoing
                return -1

        if cross_lines > 0:
            return self.CROSS
        elif circle_lines > 0:
            return self.CIRCLE
        else:
            raise Exception("Logic Error!")

    def get_action_list(self, state):
        xy_valids = np.where(state["board"][2] == 1)
        return [self.xy_2_action_id(x, y) for x, y in zip(*xy_valids)]

    def action_id_2_xy(self, action_id):
        return (action_id // 3, action_id % 3)

    def xy_2_action_id(self, x, y):
        return 3 * x + y

    def user_input_2_action(self):
        print("Please provide your next turn through numkeypad (bottom left 1)!")
        num = int(input())
        num = num - 1 if 4 <= num <= 6 else (num - 1 + 6) % 12
        x, y = self.action_id_2_xy(num)
        return self.xy_2_action_id(x, y)

    def action_2_user_output(self, action_id):
        return str(
            action_id + 1 if 4 <= action_id + 1 <= 6 else (action_id + 1 + 6) % 12
        )


def get_model():
    input = layers.Input((9 * 3,))
    tensor = input
    for _ in range(5):
        tensor = layers.Dense(units=32, use_bias=False)(tensor)
        tensor = layers.BatchNormalization(scale=False)(tensor)
        tensor = layers.Activation("relu")(tensor)
    policy_logits = layers.Dense(
        units=9, kernel_initializer="zeros", bias_initializer="zeros"
    )(tensor)
    policy = layers.Softmax()(policy_logits)
    value_logits = layers.Dense(
        units=2, kernel_initializer="zeros", bias_initializer="zeros"
    )(tensor)
    values = layers.Softmax()(value_logits)
    return tf.keras.Model(
        inputs=[input], outputs=[policy, values, policy_logits, value_logits]
    )


if __name__ == "__main__":
    from glob import glob

    stage_nbr = 0
    # mode = "train_model"
    # mode = "test_model"
    # mode = "selfplay_model"
    mode = "test_model_by_human"
    game = TicTacToe()

    if mode == "train_model":
        from tools.write_parse_tfrecords import (
            get_filenames_and_feature_format,
            tfrecord_parser,
        )
        from utils import get_time_stamp
        import functools
        import matplotlib.pyplot as plt

        network = get_model()
        network.summary(line_length=120)
        for stage in range(stage_nbr + 1):
            train_data_dir = osp.join(
                os.getenv("OUTPUT_DATADIR"),
                game.__class__.__name__,
                "trainsamples_selfplay",
                "stage_%d" % stage,
            )
            fnames, feature_format = get_filenames_and_feature_format(train_data_dir)
            np.random.shuffle(fnames)
            valid_fnames = fnames[: len(fnames) // 10]
            train_fnames = fnames[len(fnames) // 10 :]
            tf_train_dataset = tfrecord_parser(train_fnames, feature_format).batch(32)
            tf_valid_dataset = tfrecord_parser(valid_fnames, feature_format).batch(32)
            train_iterator = tf_train_dataset.make_one_shot_iterator()
            valid_iterator = tf_valid_dataset.make_one_shot_iterator()
            inputs = train_iterator.get_next()
            valid_inputs = valid_iterator.get_next()
            flatten_state = layers.Input(
                shape=inputs["flatten_state"].shape.as_list()[1:]
            )
            outputs = network(flatten_state)
            model = tf.keras.Model(inputs=flatten_state, outputs=outputs)
            logits_xentropy = functools.wraps(tf.keras.losses.categorical_crossentropy)(
                functools.partial(
                    tf.keras.losses.categorical_crossentropy, from_logits=True
                )
            )
            model.compile(
                optimizer="adam",
                loss=[
                    lambda *args: tf.zeros_like(args[0]),
                    lambda *args: tf.zeros_like(args[0]),
                    logits_xentropy,
                    logits_xentropy,
                ],
                metrics=[["accuracy", "mean_squared_error"], ["accuracy"], [], []],
            )
            history = model.fit(
                x=inputs["flatten_state"],
                y=[
                    inputs["policy_label"],
                    inputs["values_label"],
                    inputs["policy_label"],
                    inputs["values_label"],
                ],
                validation_data=(
                    valid_inputs["flatten_state"],
                    [
                        valid_inputs["policy_label"],
                        valid_inputs["values_label"],
                        valid_inputs["policy_label"],
                        valid_inputs["values_label"],
                    ],
                ),
                steps_per_epoch=len(train_fnames) // 32,
                validation_steps=len(valid_fnames) // 32,
                epochs=100,
                verbose=2,
            )
            timestamp = get_time_stamp()
            model_filename = osp.join(
                os.getenv("INPUT_DATADIR"),
                "models",
                game.__class__.__name__,
                "stage_%d" % stage_nbr,
                timestamp + "_trainstage_%d.h5" % stage,
            )
            os.makedirs(osp.dirname(model_filename), exist_ok=True)
            network.save(model_filename)
            plt.plot(history.history["model_acc"], label="policy_acc")
            plt.plot(history.history["val_model_acc"], label="val_policy_acc")
            plt.plot(
                -np.log(np.array(history.history["model_mean_squared_error"])),
                label="policy_-ln_mse",
            )
            plt.plot(
                -np.log(np.array(history.history["val_model_mean_squared_error"])),
                label="val_policy_-ln_mse",
            )
            plt.plot(history.history["model_1_acc"], label="values_acc")
            plt.plot(history.history["val_model_1_acc"], label="val_values_acc")
            plt.legend()
            plt.savefig(model_filename[:-2] + "png")
            plt.close("all")
    elif mode == "selfplay_model":
        if stage_nbr == 0:
            player = MCTS_RandomPolicy(
                game,
                train_data_output_dir=osp.join(
                    os.getenv("OUTPUT_DATADIR"),
                    game.__class__.__name__,
                    "trainsamples_selfplay",
                    "stage_%d" % stage_nbr,
                ),
                temperature=1.0,
                c_puct=20.0,
                nbr_sims=2 ** 8,
            )
        else:
            model_files = sorted(
                glob(
                    osp.join(
                        os.getenv("INPUT_DATADIR"),
                        "models",
                        game.__class__.__name__,
                        "stage_%d" % (stage_nbr - 1),
                        "*_trainstage_%d.h5" % (stage_nbr - 1),
                    )
                )
            )
            network = tf.keras.models.load_model(model_files[-1])
            network.summary(line_length=120)
            player = MCTS_NetworkPolicy(
                network,
                game,
                train_data_output_dir=osp.join(
                    os.getenv("OUTPUT_DATADIR"),
                    game.__class__.__name__,
                    "trainsamples_selfplay",
                    "stage_%d" % stage_nbr,
                ),
                temperature=1.0,
                c_puct=20.0,
                nbr_sims=2 ** 8,
            )

        players = [player, player]
        n = 1000
        sum_points = np.array([0.0, 0.0])
        for _ in tqdm(range(n)):
            engine = GameEngine(game, players)
            result = engine.run()
            sum_points += np.array(result)
            print(sum_points / sum_points.sum())
    elif mode == "test_model":
        model_files = sorted(
            glob(
                osp.join(
                    os.getenv("INPUT_DATADIR"),
                    "models",
                    game.__class__.__name__,
                    "stage_%d" % stage_nbr,
                    "*_trainstage_%d.h5" % stage_nbr,
                )
            )
        )
        network = tf.keras.models.load_model(model_files[-1])
        network.summary(line_length=120)
        if stage_nbr == 0:
            older_network_generation = get_model()
        else:
            model_files = sorted(
                glob(
                    osp.join(
                        os.getenv("INPUT_DATADIR"),
                        "models",
                        game.__class__.__name__,
                        "stage_%d" % (stage_nbr - 1),
                        "*_trainstage_%d.h5" % (stage_nbr - 1),
                    )
                )
            )
            older_network_generation = tf.keras.models.load_model(model_files[-1])

        players = [
            MCTS_NetworkPolicy(
                network, game, temperature=0.0, c_puct=20.0, nbr_sims=2 ** 5
            ),
            MCTS_NetworkPolicy(
                older_network_generation,
                game,
                temperature=0.0,
                c_puct=20.0,
                nbr_sims=2 ** 5,
            ),
        ]
        n = 100
        sum_points = np.array([0.0, 0.0])
        for _ in tqdm(range(n)):
            engine = GameEngine(game, players)
            result = engine.run()
            sum_points += np.array(result)
            print("Current win estimates:", sum_points / sum_points.sum())
    elif mode == "test_model_by_human":
        model_files = sorted(
            glob(
                osp.join(
                    os.getenv("INPUT_DATADIR"),
                    "models",
                    game.__class__.__name__,
                    "stage_%d" % stage_nbr,
                    "*_trainstage_%d.h5" % stage_nbr,
                )
            )
        )
        network = tf.keras.models.load_model(model_files[-1])
        network.summary(line_length=120)

        players = [
            MCTS_NetworkPolicy(
                network, game, temperature=0.0, c_puct=20.0, nbr_sims=2 ** 5
            ),
            # MCTS_RandomPolicy(game, temperature=0.0, c_puct=20.0, nbr_sims=2 ** 3),
            # MCTS_RandomPolicy(game, temperature=0.0, c_puct=20.0, nbr_sims=2 ** 10),
            Human(game),
        ]
        n = 3
        sum_points = np.array([0.0, 0.0])
        for _ in tqdm(range(n)):
            engine = GameEngine(game, players)
            result = engine.run()
            sum_points += np.array(result)
        print(sum_points / sum_points.sum())
