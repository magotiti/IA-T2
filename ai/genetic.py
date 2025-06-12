import random
import numpy as np
from utils.config import *
from ai.neural_net import NeuralNet
from game.board import Board
from game.minimax import get_minimax_move

# cronograma de dificuldade
EASY_PHASE = 3      # geracoes 1 a 3
MEDIUM_PHASE = 3    # geracoes 4 a 6
# geracoes 7 em diante sao hard

def current_difficulty(gen: int) -> str:
    if gen <= EASY_PHASE:
        return "easy"
    elif gen <= EASY_PHASE + MEDIUM_PHASE:
        return "medium"
    return "hard"


class GeneticAlgorithm:
    def __init__(self):
        # numero de pesos da rede
        self.genes = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE
        self.population = [
            np.random.uniform(-1, 1, self.genes) for _ in range(POPULATION_SIZE)
        ]
        self.generation = 0
        self.current_diff = current_difficulty(self.generation)

    # avaliacao da geracao
    def evaluate(self):
        self.generation += 1
        self.current_diff = current_difficulty(self.generation)

        scores = []
        for idx, chrom in enumerate(self.population):
            try:
                score = self.fitness(chrom)
                if score is not None:
                    scores.append((chrom, score))
            except Exception as e:
                print(f"erro cromossomo {idx}: {e}")

        if not scores:
            raise RuntimeError("nenhum cromossomo valido")

        scores.sort(key=lambda x: x[1], reverse=True)
        self.population = [chrom for chrom, _ in scores[:POPULATION_SIZE]]

        best_score = scores[0][1]
        self.best_fitness = best_score
        return best_score

    # funcao de aptidao
    def fitness(self, weights):
        try:
            net = NeuralNet(weights)
            board = Board()
            player = 1  # rede comeca
            penalties = 0
            move_count = 0

            while True:
                available = board.available_moves()
                if not available:
                    return penalties + 3  # empate

                if player == 1:
                    output = net.predict(board.state)
                    mask = [output[i] if i in available else -np.inf for i in range(9)]
                    move = int(np.argmax(mask))
                else:
                    move = get_minimax_move(board, player, self.current_diff)

                if move is None or move not in available:
                    return penalties - 5  # movimento invalido

                if not board.make_move(move, player):
                    return penalties - 5

                move_count += 1
                winner = board.check_winner()
                if winner is not None:
                    if winner == 1:
                        bonus = 9 - move_count  # bonus por vitoria rapida
                        return penalties + 10 + bonus
                    if winner == 0:
                        return penalties + 3
                    phase_penalty = 0
                    if self.current_diff == "medium":
                        phase_penalty = -5
                    elif self.current_diff == "hard":
                        phase_penalty = -10
                    return penalties + phase_penalty

                player *= -1
        except Exception as e:
            print(f"erro fitness: {e}")
            return -9999

    # gera proxima geracao
    def next_generation(self):
        new_pop = []
        new_pop.extend(self.population[:2])  # elitismo

        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(self.population[:10], 2)
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2

            # mutacao gaussiana
            m = np.random.rand(*child.shape) < MUTATION_RATE
            child[m] += np.random.normal(0, 0.1, np.sum(m))

            new_pop.append(child)

        self.population = new_pop


__all__ = ["GeneticAlgorithm"]
