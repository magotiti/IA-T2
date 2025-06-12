import random
from functools import lru_cache

# utilidades basicas sobre uma state de 9 casas

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),          # linhas
    (0, 3, 6), (1, 4, 7), (2, 5, 8),          # colunas
    (0, 4, 8), (2, 4, 6),                     # diagonais
]

def winner_state(state):
    """devolve 1 (x), -1 (o), 0 (empate) ou none (continua)."""
    for a, b, c in WIN_LINES:
        if state[a] == state[b] == state[c] != 0:
            return state[a]               # 1 ou -1
    if 0 not in state:
        return 0                          # empate
    return None                           # jogo em aberto


# minimax perfeito com memoizacao

@lru_cache(maxsize=None)
def minimax_state(state_tuple, player):
    """
    calcula (melhor_score) a partir de uma tupla hasheavel (len=9)
    e do player (1 = x, -1 = o). score sempre relativo ao player
    que chamou a funcao.
    """
    result = winner_state(state_tuple)
    if result is not None:                # terminal
        return result * player            # vitoria(1) empate(0) derrota(-1)

    best = -2                             # pior possivel
    state = list(state_tuple)
    for i in range(9):
        if state[i] == 0:
            state[i] = player
            score = -minimax_state(tuple(state), -player)   # ponto de vista adversario
            state[i] = 0
            best = max(best, score)
            if best == 1:                 # poda alfa beta implicita
                break
    return best


def minimax_move(state, player):
    """retorna o indice da melhor jogada para player em state."""
    best_moves, best_score = [], -2
    for i in range(9):
        if state[i] == 0:
            new_state = list(state)
            new_state[i] = player
            score = -minimax_state(tuple(new_state), -player)
            if score > best_score:
                best_score = score
                best_moves = [i]
            elif score == best_score:
                best_moves.append(i)
    return random.choice(best_moves)       # desempate aleatorio


# wrappers por dificuldade (easy / medium / hard)

def minimax_easy(board, player):
    """25 por cento minimax, 75 por cento aleatorio."""
    return minimax_move(board.state, player) if random.random() < 0.25 else random.choice(board.available_moves())

def minimax_medium(board, player):
    """50 por cento minimax, 50 por cento aleatorio."""
    return minimax_move(board.state, player) if random.random() < 0.50 else random.choice(board.available_moves())

def minimax_hard(board, player):
    """100 por cento jogo perfeito."""
    return minimax_move(board.state, player)


# interface unica para o ga / cli

def get_minimax_move(board, player, difficulty="easy"):
    d = difficulty.lower()
    if d == "easy":
        return minimax_easy(board, player)
    if d == "medium":
        return minimax_medium(board, player)
    if d == "hard":
        return minimax_hard(board, player)
    raise ValueError(f"unknown difficulty: {difficulty}")


__all__ = [
    "minimax_easy", "minimax_medium", "minimax_hard", "get_minimax_move"
]
