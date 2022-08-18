import numpy as np


class mlp:
    def __init__(self, input_size, layers=[10, 10, 10, 1], learning_rate=0.2, epochs=10):
        
        # dataset attributes
        self.input_size = input_size
        self.num_classes = layers[-1]

        # model parameters
        self.weights = []
        self.biases = []
        self.z = []
        self.h = []
        self.gradients_weights = []
        self.gradients_biases = []

        # hyper-parameters
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs


    def _build(self):
        
        # lets initialize the weights, bias and gradients
        prev = None
        for layer in self.layers:
            if prev:
                self.weights.append(0.001 * np.random.randn(prev, layer))
                self.gradients_weights.append(np.zeros([prev, layer]))
            else:
                self.weights.append(0.001 * np.random.randn(self.input_size, layer))
                self.gradients_weights.append(np.zeros([self.input_size, layer]))
            self.biases.append(np.zeros(layer))
            self.gradients_biases.append(np.zeros(layer))
            prev = layer
        
        self.nn = []
        for layer in self.layers:
            self.nn.append(layer)


    def fit(self, x, y):
        self._build()
        
        for epoch in range(self.epochs):
            print(epoch)
            self.forward(x, y)
    
    
    def forward(self, x, y, train=True):
        
        # forward path
        prev = None
        self.h = []
        self.z = []
        for layer, weight, bias in zip(self.layers, self.weights, self.biases):
            if prev is not None:
                z = np.dot(prev, weight) + bias
            else:
                z = np.dot(x, weight) + bias
            h = self.sigmoid(z)
            print(z, h)
            self.z.append(z)
            self.h.append(h)
            prev = h
        prob = self.softmax(h)
        loss = self.binary_cross_entropy_loss(prob, y)
        
        if train:
            # backwards path
            prev = None
            for i in reversed(range(len(self.gradients_weights))):
                if prev is not None:
                    error = np.dot(self.weights[i], prev) * self.sigmoid_dev(self.z[i])
                else:
                    error = (y - prob) * self.sigmoid_dev(self.z[i])
                print(self.h[i], error.T)
                self.gradients_weights[i] = np.dot(error.T, self.h[i])
                self.gradients_biases[i] = error
                
                print(self.gradients_biases[i], self.weights[i])
                self.weights[i] += self.learning_rate * self.gradients_weights[i]
                self.biases[i] += self.learning_rate * self.gradients_biases[i]
                
                prev = error

    def softmax(self, scores):
        exp = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        sum_exp = np.sum(exp, axis=-1, keepdims=True)
        prob =  exp / sum_exp
        return prob
        
        
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-1. * x))
    
    
    def sigmoid_dev(self, x):
        sigmoid = self.sigmoid(x)
        ds = sigmoid * (1-sigmoid)
        return ds


    def binary_cross_entropy_loss(self, pred, y):
        # get data shape
        m = pred.shape[0]

        cross_entropy = y * np.log(pred)
        cost = -np.sum(cross_entropy)
        cost = cost / m
        # remove single deminsional entries from shape
        loss = np.squeeze(cost)
        
        return loss
