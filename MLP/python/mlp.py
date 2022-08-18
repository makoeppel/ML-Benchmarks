import numpy as np
from sklearn.metrics import roc_auc_score


class mlp:
    def __init__(self, input_size, layers=[3, 1], learning_rate=0.1, epochs=5000):
        
        # dataset attributes
        self.input_size = input_size
        self.num_classes = layers[-1]

        # model parameters
        self.weights = []
        self.biases = []
        self.z = []
        self.s = []

        # hyper-parameters
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs


    def _build(self):
        
        # lets initialize the weights, bias and gradients
        prev = None
        for layer in self.layers:
            if prev is not None:
                self.weights.append(np.random.uniform(size=(prev, layer)))
            else:
                self.weights.append(np.random.uniform(size=(self.input_size, layer)))
            self.biases.append(np.random.uniform(size=(1, layer)))
            prev = layer


    def fit(self, x, y):
        self._build()
        
        errors = []
        for epoch in range(self.epochs):
            
            error, y_pred = self.forward(x, y)
            auc = roc_auc_score(y, y_pred)
            print(f"Epoch {epoch} - Error {error} - AUC {auc}")
            errors.append(error)
        return errors


    def nn_layer(self, w, x, b):
        return (x @ w) + b


    def predict(self, x):
        
        prev = None
        self.z = []
        self.s = []
        for w, b in zip(self.weights, self.biases):
            if prev is not None:
                z = self.nn_layer(w, prev, b)
                s = self.sigmoid(z)
            else:
                z = self.nn_layer(w, x, b)
                s = self.sigmoid(z)
            self.z.append(z)
            self.s.append(s)
            prev = s

        return s


    def backwards(self, x, y):
        
        deltaPrev = None
        #for j, s in enumerate(reversed(self.s), start=1):
        for idx in reversed(range(len(self.s))):
            # calculate gradients
            if deltaPrev is None:
                # start with the last value of self.s
                delta = (self.s[idx] - y) * self.s[idx] * (1 - self.s[idx])
                gradients = self.s[idx-1].T @ delta
            elif idx == 0:
                # for the end we need to take the input data x
                delta = (deltaPrev @ self.weights[idx+1].T) * self.s[idx] * (1 - self.s[idx])
                gradients = x.T @ delta
            else:
                delta = (deltaPrev @ self.weights[idx+1].T) * self.s[idx] * (1 - self.s[idx])
                gradients = self.s[idx-1].T @ delta
            
            # update parameters
            self.weights[idx] -= self.learning_rate * gradients
            self.biases[idx] -= self.learning_rate * np.sum(delta, axis=0, keepdims=True)
            
            deltaPrev = delta

    
    def forward(self, x, y, train=True):
        
        s = self.predict(x)
        
        if train:
            # compute error
            error = self.squard_error(s, y)

            # update parameters
            self.backwards(x, y)

        return error, s


    def squard_error(self, A, y):
        return (np.mean(np.power(A - y, 2)))/2
        

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

