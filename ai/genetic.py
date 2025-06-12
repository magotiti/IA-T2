import random
import numpy as np
from utils.config import *
from ai.neural_net import NeuralNet
from game.board import Board
from game.minimax import minimax_easy

class GeneticAlgorithm:
    def __init__(self):
        self.genes = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE
        self.population = [np.random.uniform(-1, 1, self.genes) for _ in range(POPULATION_SIZE)]

    def evaluate(self):
        scores = []
        for idx, chrom in enumerate(self.population):
            try:
                score = self.fitness(chrom)
                if score is not None:
                    scores.append((chrom, score))
                else:
                    print(f"[Atenção] Cromossomo {idx} retornou None.")
            except Exception as e:
                print(f"[Erro] Durante avaliação do cromossomo {idx}: {e}")

        if not scores:
            print("[Erro Crítico] Nenhum cromossomo avaliado com sucesso.")
            exit(1)

        scores.sort(key=lambda x: x[1], reverse=True)
        self.population = [chrom for chrom, _ in scores[:POPULATION_SIZE]]

    def fitness(self, weights):
        try:
            net = NeuralNet(weights)
            board = Board()
            board.reset()
            player = 1
            penalties = 0

            while True:
                available = board.available_moves()

                if not available:
                    return penalties + 3  # empate

                if player == 1:
                    output = net.predict(board.state)
                    mask = [output[i] if i in available else -np.inf for i in range(9)]
                    move = int(np.argmax(mask))
                else:
                    move = minimax_easy(board, player)

                if move is None or move not in available:
                    penalties -= 5
                    return penalties

                valid = board.make_move(move, player)
                if not valid:
                    penalties -= 5
                    return penalties

                winner = board.check_winner()
                if winner is not None:
                    return penalties + (10 if winner == 1 else 3 if winner == 0 else 0)

                player *= -1

        except Exception as e:
            print(f"[Erro na função fitness]: {e}")
            return -9999  # penalidade bruta
        

    def next_generation(self):
        new_pop = []
        elite = self.population[:2]
        new_pop.extend(elite)

        while len(new_pop) < POPULATION_SIZE:
            p1, p2 = random.sample(self.population[:10], 2)
            alpha = np.random.rand()
            child = alpha * p1 + (1 - alpha) * p2

            # Mutação
            mutation = np.random.rand(*child.shape) < MUTATION_RATE
            child[mutation] += np.random.normal(0, 0.1, np.sum(mutation))

            new_pop.append(child)

        self.population = new_pop
