from mlp import mlp
import numpy as np

x = np.genfromtxt('../../data/breast.csv', delimiter=",")
y = np.genfromtxt('../../data/breast_y.csv', delimiter=",")

cls = mlp(len(x[0]))
cls.fit(x, y)
