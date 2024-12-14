import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from utils import tanh, tanh_prime, mse, mse_prime


net = Network()
net.add(FCLayer(4, 16))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(16, 3))
net.add(ActivationLayer(tanh, tanh_prime))