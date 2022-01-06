import numpy as np


class Game:
    def __init__(self):
        self.p1 = np.zeros((3, 3), dtype=bool)
        self.p2 = np.zeros((3, 3), dtype=bool)

    def make_move_skip_validation(self, player: int, move: int):
        """
        Makes a move for the given player, but doesn't check if that move is allowed
        :param player: The player to make a move for. An integer, either 1 or 2
        :param move: The position to place the piece. An integer, 1 through 9, left to right then top to bottom
        :return: None
        """
        if player == 1:
            self.p1[move//3, move%3] = True
        elif player == 2:
            self.p2[move] = True

    def valid_moves(self):
        moves = ~ (self.p1 | self.p2)
        return set(np.where(moves.flatten())[0])

    def is_winner(self):
        return (
            3 in self.p1.sum(axis=0) |
            3 in self.p1.sum(axis=1) |
            3 == self.p1.trace()
        ), (
            3 in self.p2.sum(axis=0) |
            3 in self.p2.sum(axis=1) |
            3 == self.p2.trace()
        )

    def players_turn(self):
        return 1 if sum(self.p1) == sum(self.p2) else 2

    def game_finished(self):
        return True not in self.valid_moves()

    def get_boards(self):
        return self.p1, self.p2

    def __str__(self):
        return str([self.p1, self.p2])


def main():
    g = Game()
    g.make_move_skip_validation(1, 6)
    merged = np.concatenate((g.p1, g.p2), axis=1)
    print(merged.shape)
    print(repr(merged))
    print(g.valid_moves())


if __name__ == '__main__':
    main()


