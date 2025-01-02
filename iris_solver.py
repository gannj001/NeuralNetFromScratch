import numpy as np
import pandas as pd

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from utils import tanh, tanh_prime, mse, mse_prime

data = pd.read_csv("./iris_dataset/Iris.csv")
training_count = int(len(data.index)*9/10)
training_data = data.sample(training_count)


net = Network()
net.add(FCLayer(4, 16))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(16, 3))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
