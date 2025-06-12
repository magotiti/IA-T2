import random

def minimax_easy(board, player):
    available = board.available_moves()
    if not available:
        return None

    if random.random() < 0.25:
        move = minimax(board, player)
        return move if move in available else random.choice(available)

    return random.choice(available)

def minimax(board, player):
    winner = board.check_winner()
    if winner is not None:
        return None, winner * player

    best_score = -float('inf') if player == 1 else float('inf')
    best_move = None

    for move in board.available_moves():
        board.make_move(move, player)
        _, score = minimax(board, -player)
        board.state[move] = 0

        if player == 1 and score > best_score:
            best_score = score
            best_move = move
        elif player == -1 and score < best_score:
            best_score = score
            best_move = move

    if best_move is None:
        return random.choice(board.available_moves()), best_score

    return best_move, best_score
