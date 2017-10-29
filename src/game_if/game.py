#!/user/bin/env python

'''game.py: Provides the game class interface for implementing full information games.'''

################################################################################

class Game:

    def __init__(self):
        self.nbr_players = 2

    def take_action(player_id, action_id):
        return player_id_who_has_won #(-1 if invalid move, -2 if still ongoing, -3 if draw)

    def get_action_list():
        return
