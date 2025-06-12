import numpy as np
from utils.config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

class NeuralNet:
    def __init__(self, weights):
        self.w1 = weights[:INPUT_SIZE * HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
        self.w2 = weights[INPUT_SIZE * HIDDEN_SIZE:].reshape(HIDDEN_SIZE, OUTPUT_SIZE)

    def predict(self, inputs):
        x = np.array(inputs)
        h = np.tanh(np.dot(x, self.w1))
        o = np.dot(h, self.w2)
        return o