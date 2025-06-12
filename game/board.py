import numpy as np

class Board:
    def __init__(self):
        self.state = np.zeros(9, dtype=int)  # 0 vazio, 1 X, -1 O

    def reset(self):
        self.state[:] = 0

    def display(self):
        symbols = {1: 'X', -1: 'O', 0: ' '}
        print("\nTabuleiro:")
        for i in range(0, 9, 3):
            print('|'.join(symbols[self.state[i + j]] for j in range(3)))
        print()

    def available_moves(self):
        return [i for i in range(9) if self.state[i] == 0]

    def make_move(self, pos, player):
        if self.state[pos] == 0:
            self.state[pos] = player
            return True
        return False

    def check_winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8),
                (0,3,6), (1,4,7), (2,5,8),
                (0,4,8), (2,4,6)]
        for i,j,k in wins:
            line_sum = self.state[i] + self.state[j] + self.state[k]
            if line_sum == 3:
                return 1
            elif line_sum == -3:
                return -1
        if 0 not in self.state:
            return 0  # empate
        return None  # jogo em andamento