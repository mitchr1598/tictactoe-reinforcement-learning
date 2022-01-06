import numpy as np
from game import Game

class DataGenerator:
    def __init__(self, network):
        self.network = network
        self.results = []
        self.x, self.y = None, None
    
    def sim_games(self, n):
        self.results = [self.run_game() for _ in range(n)]
        self.x = np.stack((d1 for d1, _ in self.results), axis=0)
        self.y = np.stack((d2 for _, d2 in self.results), axis=0)  # Needs to be converted  to desired outputs (maybe 1 and 0 is ok)

    def run_game(self):
        g = Game()
        new_data = []
        while not g.game_finished():
            b1, b2 = g.get_boards()
            player = g.players_turn()
            best_move = self.network.choose_move(b1, b2)
            new_data.append((b1, b2, player, best_move))
            g.make_move_skip_validation(player, best_move)
        winner = 1 if g.is_winner()[0] else 2
        new_data_with_winner = [t + (t[2]==winner,) for t in new_data]
        return (
            np.array([np.concatenate((p1, p2), axis=1) for p1, p2, _, _, _ in new_data_with_winner]),  # The new x data
            np.array([win for _, _, _, _, win in new_data_with_winner])  # boolean on if won, to be converted to y data
        )
        




