from mlp import mlp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler


# get data
toy = False
if toy:
    # XOR
    x = np.array([  [0, 0, 1, 1],
                    [0, 1, 0, 1]]).T
    y = np.array([[0, 1, 1, 0]]).T
else:
    x = np.genfromtxt('../../data/breast.csv', delimiter=",")
    y = np.genfromtxt('../../data/breast_y.csv', delimiter=",")
    x = MinMaxScaler().fit_transform(x)

# train the model
if toy:
    cls = mlp(input_size=len(x[0]), layers=[3, 1], learning_rate=0.1, epochs=5000)
else:
    cls = mlp(input_size=len(x[0]), layers=[40, 20, 10, 1], learning_rate=0.0001, epochs=10000)
errors = cls.fit(x, y.reshape(-1,1))

# evaluation
y_pred = cls.predict(x)
y_pred = np.where(y_pred >= 0.5, 1, 0)
num_correct_predictions = (y_pred == y).sum()
accuracy = (num_correct_predictions / y.shape[0]) * 100
print('Multi-layer perceptron accuracy: %.2f%%' % accuracy)

# plot training loss
plt.plot(errors, label="train")
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.legend()
plt.savefig("python-train-mse.pdf")
plt.savefig("python-train-mse.png")
