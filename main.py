from ai.genetic import GeneticAlgorithm

def main():
    ga = GeneticAlgorithm()
    NUM_GERACOES = 10

    for _ in range(NUM_GERACOES):
        melhor = ga.evaluate()          # devolve melhor aptidao
        print(
            f"geracao {ga.generation} | "
            f"dificuldade {ga.current_diff} | "
            f"melhor aptidao: {melhor}"
        )
        ga.next_generation()

if __name__ == "__main__":
    main()
