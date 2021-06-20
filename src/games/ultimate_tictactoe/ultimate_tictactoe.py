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


class UltimateTicTacToe(Game):
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
    state_dimensionality = 81 * 3 + 9
    max_nbr_actions = 81

    def generate_initial_state(self):
        board = np.zeros(shape=(3, 3, 3, 3, 3))
        superboard = np.zeros(shape=(3, 3, 3))
        board[2, :, :, :, :] = 1
        superboard[2, :, :] = 1
        return dict(
            board=board, superboard=superboard, turn=self.CIRCLE, last_move=None
        )

    def flatten_state(self, state):
        lm = np.zeros(9)
        if state["last_move"] is not None and state["last_move"] >= 0:
            lm[state["last_move"] % 9] = 1.0
        if state["turn"] == 0:
            return np.concatenate([state["board"].flatten(), lm])
        elif state["turn"] == 1:
            return np.concatenate([state["board"][[1, 0, 2]].flatten(), lm])
        else:
            raise Exception("The player %d does not exist!" % state["turn"])

    def visualize(self, state):
        xorder = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        yorder = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ords = ord(" ") * state["board"][2, xorder, yorder, :, :]
        ords += ord("O") * state["board"][0, xorder, yorder, :, :]
        ords += ord("X") * state["board"][1, xorder, yorder, :, :]
        ords = np.array(ords, dtype=int)
        ords[
            np.logical_and(
                np.logical_and(ords != ord(" "), ords != ord("X")), ords != ord("O")
            )
        ] = ord("#")
        lines = []
        lines.append("---------------------------------------")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[0:3, 0, :].flatten()])
        )
        lines.append("---+---+---   ---+---+---   ---+---+---")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[0:3, 1, :].flatten()])
        )
        lines.append("---+---+---   ---+---+---   ---+---+---")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[0:3, 2, :].flatten()])
        )
        lines.append("                                       ")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[3:6, 0, :].flatten()])
        )
        lines.append("---+---+---   ---+---+---   ---+---+---")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[3:6, 1, :].flatten()])
        )
        lines.append("---+---+---   ---+---+---   ---+---+---")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[3:6, 2, :].flatten()])
        )
        lines.append("                                       ")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[6:9, 0, :].flatten()])
        )
        lines.append("---+---+---   ---+---+---   ---+---+---")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[6:9, 1, :].flatten()])
        )
        lines.append("---+---+---   ---+---+---   ---+---+---")
        lines.append(
            " %s | %s | %s     %s | %s | %s     %s | %s | %s "
            % tuple([chr(s) for s in ords[6:9, 2, :].flatten()])
        )
        lines.append("---------------------------------------")
        return "\n".join(lines)

    def minimal_visualize(self, state):
        xorder = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        yorder = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        ords = ord(" ") * state["board"][2, xorder, yorder, :, :]
        ords += ord("O") * state["board"][0, xorder, yorder, :, :]
        ords += ord("X") * state["board"][1, xorder, yorder, :, :]
        ords = np.array(ords, dtype=int)
        ords[
            np.logical_and(
                np.logical_and(ords != ord(" "), ords != ord("X")), ords != ord("O")
            )
        ] = ord("#")
        lines = []
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[0:3, 0, :].flatten()])
        )
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[0:3, 1, :].flatten()])
        )
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[0:3, 2, :].flatten()])
        )
        lines.append("-----------")
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[3:6, 0, :].flatten()])
        )
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[3:6, 1, :].flatten()])
        )
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[3:6, 2, :].flatten()])
        )
        lines.append("-----------")
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[6:9, 0, :].flatten()])
        )
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[6:9, 1, :].flatten()])
        )
        lines.append(
            "%s%s%s|%s%s%s|%s%s%s" % tuple([chr(s) for s in ords[6:9, 2, :].flatten()])
        )
        return "\n".join(lines)

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

        xs, ys, x, y = self.action_id_2_xy(action_id)

        board = state["board"].copy()
        if action_id not in self.get_action_list(state):
            last_move = -1
        else:
            assert state["superboard"][2, xs, ys] == 1.0
            assert state["board"][2, xs, ys, x, y] == 1.0
            last_move = action_id

            board[2, xs, ys, x, y] -= 1
            board[player_id, xs, ys, x, y] += 1

        res = self.evaluate_subboard(board[:, xs, ys, :, :])
        if res == -2 and state["superboard"][:, xs, ys].sum() == 1:
            superboard = state["superboard"].copy()
            superboard[2, xs, ys] -= 1
        elif res >= 0 and state["superboard"][res, xs, ys] != 1.0:
            superboard = state["superboard"].copy()
            superboard[2, xs, ys] -= 1
            superboard[res, xs, ys] += 1
        else:
            superboard = state["superboard"]

        return dict(
            board=board,
            superboard=superboard,
            turn=1 - state["turn"],
            last_move=last_move,
        )

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
        assert np.max(state["board"][2, :, :, :, :]) <= 1
        assert np.min(state["board"][:2, :, :, :, :]) >= 0
        assert np.all(np.sum(state["board"], axis=0) == np.ones([3, 3, 3, 3]))

        if np.min(state["board"][2, :, :, :, :]) < 0:  # invalid move
            return -3
        if state["last_move"] is not None and state["last_move"] == -1:
            # also invalid move, because move to invalid global square
            return -3

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= state["superboard"][self.CIRCLE][cell]
                cros_f *= state["superboard"][self.CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert cross_lines * circle_lines == 0

        if cross_lines == 0 and circle_lines == 0:
            # no possible move left, it is a draw
            if np.max(state["superboard"][2, :, :]) == 0:
                return -2
            else:  # game is still ongoing
                return -1

        if cross_lines > 0:
            return self.CROSS
        elif circle_lines > 0:
            return self.CIRCLE
        else:
            raise Exception("Logic Error!")

    def evaluate_subboard(self, subboard):
        assert np.max(subboard[2, :, :]) <= 1
        assert np.min(subboard[:2, :, :]) >= 0
        assert np.all(np.sum(subboard, axis=0) == np.ones([3, 3]))

        if np.min(subboard[2, :, :]) < 0:  # invalid move
            raise Exception("LogicError")

        circle_lines = 0
        cross_lines = 0

        for line in self.lines:
            circ_f = 1
            cros_f = 1
            for cell in line:
                circ_f *= subboard[self.CIRCLE][cell]
                cros_f *= subboard[self.CROSS][cell]
            circle_lines += circ_f
            cross_lines += cros_f

        assert cross_lines * circle_lines == 0

        if cross_lines == 0 and circle_lines == 0:
            if np.max(subboard[2, :, :]) == 0:  # no possible move left, it is a draw
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
        poss_actions = []
        xss, yss = (-1, -1)
        if state["last_move"] is not None and state["last_move"] >= 0:
            lxs, lys, xss, yss = self.action_id_2_xy(state["last_move"])
            if state["superboard"][2, xss, yss] == 0.0:
                xss, yss = (-1, -1)
        for x in range(3):
            for y in range(3):
                if xss == -1:
                    for xs in range(3):
                        for ys in range(3):
                            if (
                                state["board"][2, xs, ys, x, y] == 1.0
                                and state["superboard"][2, xs, ys] == 1.0
                            ):
                                poss_actions.append(self.xy_2_action_id(xs, ys, x, y))
                else:
                    assert state["superboard"][2, xss, yss] == 1.0
                    if state["board"][2, xss, yss, x, y] == 1.0:
                        poss_actions.append(self.xy_2_action_id(xss, yss, x, y))
        return poss_actions

    def action_id_2_xy(self, action_id):
        return (
            (action_id // 27) % 3,
            (action_id // 9) % 3,
            (action_id // 3) % 3,
            (action_id // 1) % 3,
        )

    def xy_2_action_id(self, xs, ys, x, y):
        return xs * 27 + ys * 9 + x * 3 + y

    def user_input_2_action(self):
        print(
            "Please provide your next turn through numkeypad (bottom left 1) "
            "input coordinates, eg 11 would be lower left most corner!"
        )
        s, local = map(
            lambda x: int(x) - 1 if 4 <= int(x) <= 6 else (int(x) - 1 + 6) % 12, input()
        )
        xs, ys, x, y = self.action_id_2_xy(9 * s + local)
        return self.xy_2_action_id(xs, ys, x, y)

    def action_2_user_output(self, action_id):
        return str(
            tuple(
                map(
                    lambda x: x + 1 if 4 <= x + 1 <= 6 else (x + 1 + 6) % 12,
                    [action_id // 9, action_id % 9],
                )
            )
        )


def get_model():
    input = layers.Input((81 * 3 + 9,))
    board = layers.Lambda(lambda x: x[:, : 81 * 3])(input)
    board = layers.Reshape((3, 9, 9))(board)
    board = layers.Permute([2, 1, 3])(board)
    board = layers.Reshape((9, 3 * 9))(board)
    last_move = layers.Lambda(lambda x: x[:, 81 * 3 :, None])(input)
    reshaped_state = layers.Concatenate(axis=-1)([board, last_move])
    tensor = reshaped_state
    for _ in range(5):
        tensor = layers.Dense(units=32, use_bias=False)(tensor)
        tensor = layers.BatchNormalization(scale=False)(tensor)
        tensor = layers.Activation("relu")(tensor)
    enc_tensor = layers.Reshape((32 * 9,))(tensor)
    for _ in range(5):
        enc_tensor = layers.Dense(units=512, use_bias=False)(enc_tensor)
        enc_tensor = layers.BatchNormalization(scale=False)(enc_tensor)
        enc_tensor = layers.Activation("relu")(enc_tensor)
    tiled_enc_tensor = layers.Lambda(lambda x: tf.tile(x[:, None, :], [1, 9, 1]))(
        enc_tensor
    )
    pos_indicator = layers.Lambda(
        lambda x: tf.tile(
            tf.constant(np.eye(9).astype(np.float32)[None, ...]), [tf.shape(x)[0], 1, 1]
        )
    )(tensor)
    tensor = layers.Concatenate(axis=-1)([tensor, tiled_enc_tensor, pos_indicator])
    for _ in range(1):
        tensor = layers.Dense(units=32, use_bias=False)(tensor)
        tensor = layers.BatchNormalization(scale=False)(tensor)
        tensor = layers.Activation("relu")(tensor)
    policy_logits = layers.Dense(
        units=9, kernel_initializer="zeros", bias_initializer="zeros"
    )(tensor)
    policy_logits = layers.Reshape((81,))(policy_logits)
    policy = layers.Softmax()(policy_logits)
    value_logits = layers.Dense(
        units=2, kernel_initializer="zeros", bias_initializer="zeros"
    )(enc_tensor)
    values = layers.Softmax()(value_logits)
    return tf.keras.Model(
        inputs=[input], outputs=[policy, values, policy_logits, value_logits]
    )


if __name__ == "__main__":
    from glob import glob
    import matplotlib.pyplot as plt

    stage_nbr = 0
    mode = "train_model"
    # mode = "test_model"
    # mode = "selfplay_model"
    # mode = "test_model_by_human"
    game = UltimateTicTacToe()

    sqrt10 = np.sqrt(10.0)
    c_puct_samples = [
        1e-3 * sqrt10,
        1e-2,
        1e-2 * sqrt10,
        1e-1,
        1e-1 * sqrt10,
        1e0,
        1e0 * sqrt10,
        1e1,
        1e1 * sqrt10,
        1e2,
    ]

    if mode == "train_model":
        from tools.write_parse_tfrecords import (
            get_filenames_and_feature_format,
            tfrecord_parser,
        )
        from utils import get_time_stamp
        import functools

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
            game_timestamps = sorted({"_".join(f.split("_")[:-1]) for f in fnames})
            np.random.shuffle(game_timestamps)
            valid_games_ts = game_timestamps[: len(game_timestamps) // 10]
            train_games_ts = game_timestamps[len(game_timestamps) // 10 :]
            valid_fnames = [
                f for f in fnames if "_".join(f.split("_")[:-1]) in valid_games_ts
            ]
            train_fnames = [
                f for f in fnames if "_".join(f.split("_")[:-1]) in train_games_ts
            ]
            np.random.shuffle(valid_fnames)
            np.random.shuffle(train_fnames)
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
                epochs=20,
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
                c_puct=2.0,
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
                c_puct=2.0,
                nbr_sims=2 ** 4,
            )

        players = [player, player]
        n = 10000
        n_per_cpuct_cell = n // len(c_puct_samples) ** 2
        assert n % len(c_puct_samples) ** 2 == 0
        sum_points = np.array([0.0, 0.0])
        sum_points_per_cpuct_cell = np.zeros(
            (len(c_puct_samples) + 1, len(c_puct_samples) + 1, 2)
        )
        for k in tqdm(range(n)):
            i = k % len(c_puct_samples)
            j = (k // len(c_puct_samples)) % len(c_puct_samples)
            c_puct_A = c_puct_samples[i]
            c_puct_B = c_puct_samples[j]
            players[0].mcts.c_puct = c_puct_A
            players[1].mcts.c_puct = c_puct_B
            engine = GameEngine(game, players)
            result = engine.run()
            sum_points += np.array(result)
            sum_points_per_cpuct_cell[i + 1, j + 1, :] = np.array(result)
        print(sum_points / sum_points.sum())
        delta_points_first_player_per_cpuct_cell = (
            sum_points_per_cpuct_cell[:, :, 0] - sum_points_per_cpuct_cell[:, :, 1]
        ) / n_per_cpuct_cell
        delta_points_first_player_per_cpuct_cell[0] += 1.0
        delta_points_first_player_per_cpuct_cell[:, 0] -= 1.0
        plt.imshow(delta_points_first_player_per_cpuct_cell, cmap=plt.cm.RdBu)
        plt.colorbar()
        plt.savefig(
            osp.join(
                os.getenv("OUTPUT_DATADIR"),
                game.__class__.__name__,
                "trainsamples_selfplay",
                "stage_%d" % stage_nbr,
                "win_rates_over_cpuct.png",
            )
        )
        plt.close("all")
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
                older_network_generation,
                game,
                temperature=0.0,
                c_puct=2.0,
                nbr_sims=2 ** 7,
            ),
            MCTS_NetworkPolicy(
                network, game, temperature=0.0, c_puct=2.0, nbr_sims=2 ** 7
            ),
        ]
        n = 100
        sum_points = np.array([0.0, 0.0])
        p = len(players)
        for i in tqdm(range(n)):
            engine = GameEngine(game, players[i % p :] + players[: i % p])
            result = engine.run()
            print(result)
            result = result[p - i % p :] + result[: p - i % p]
            print(result)
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
                network, game, temperature=0.0, c_puct=20.0, nbr_sims=2 ** 11
            ),
            # MCTS_RandomPolicy(game, temperature=0.0, c_puct=2.0, nbr_sims=2 ** 15),
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
