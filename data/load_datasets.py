from sklearn.datasets import fetch_openml, load_breast_cancer
import numpy as np

# load MNIST
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
np.savetxt("mnist.csv", X, delimiter=",")
np.savetxt("mnist_y.csv", y, delimiter=",")

# load Banks
data = load_breast_cancer()
np.savetxt("breast.csv", data.data, delimiter=",")
np.savetxt("breast_y.csv", data.target, delimiter=",")
