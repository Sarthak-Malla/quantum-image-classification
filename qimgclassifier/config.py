import os
import torch

import numpy as np

# store all the hyperparameters and constants
class Config:
    def __init__(self):
        self.num_epochs = 10
        self.batch_size = 32
        self.learning_rate = 0.001
        self.dataset = "mnist"
        self.num_classes = 10
        self.num_workers = 1

        self.dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.model_dir = os.path.join(self.dir_path, "models")
        self.model_name = "1_ry_qubit"
        self.model_path = os.path.join(self.model_dir, self.model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.seed = 42

        self.backend = "qasm_simulator"
        self.shots = 100
        self.shift = np.pi / 2

        self.n_qubits = 1
        self.quantum_layers = 1


