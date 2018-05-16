from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from classifier.fc_net import *
from data_utils.data_utils import get_CIFAR10_data
from data_utils.load_cole_data import load_cole_data
from solver import Solver
import os


# data = get_CIFAR10_data()
root = '/home/lhq/PycharmProjects/dl_numpy_lib'
data = load_cole_data(root)
for k, v in list(data.items()):
    if v is None:
        continue
    print(('%s: ' % k, v.shape))

model = FullyConnectedNet(hidden_dims=[512, 128], input_dim=64*64*3, num_classes=2)
solver = Solver(model, data,
                update_rule='sgd_momentum',
                optim_config={'learning_rate': 1e-3},
                lr_decay=0.95,
                num_epochs=100, batch_size=16,
                print_every=50,
                checkpoint_name='two_hidden_layer'
                )
solver.train()

