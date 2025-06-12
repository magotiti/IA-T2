from ai.genetic import GeneticAlgorithm
from utils.config import NUM_GENERATIONS

def main():
    ga = GeneticAlgorithm()
    for gen in range(NUM_GENERATIONS):
        print(f"\nGeração {gen+1}:")
        ga.evaluate()
        print("Melhor aptidão:", ga.fitness(ga.population[0]))
        ga.next_generation()

if __name__ == "__main__":
    main()