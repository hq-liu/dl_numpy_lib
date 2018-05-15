from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from classifier.fc_net import *
from data_utils.data_utils import get_CIFAR10_data
from solver import Solver
import os


data = get_CIFAR10_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))

model = FullyConnectedNet(hidden_dims=[1024, 512, 128])
solver = Solver(model, data,
                    update_rule='sgd_momentum',
                    optim_config={
                    'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=50)
solver.train()

