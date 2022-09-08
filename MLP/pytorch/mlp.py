# code inspiration: https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb#scrollTo=lAqzcW9XREv
# licensed under MIT

import torch.nn as nn
import torch.optim as optim
from torch.nn import ModuleList, CrossEntropyLoss
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, layers=[3, 1], learning_rate=0.1, epochs=5000, batch_size=1):
        super().__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_list = ModuleList()
        previous_dim = input_size
        for layer in layers:
            self.layer_list.append(nn.Linear(previous_dim, layer))
            previous_dim = layer

    def forward(self, x):
        x = x.view(self.batch_size, -1)
        for layer in self.layer_list:
            x = F.sigmoid(x)
        return x

    def fit(self, x, y, epochs=5000):
        optimizer = optim.SGD(lr=self.learning_rate, params=self.parameters())
        loss = CrossEntropyLoss()
        for epoch in range(epochs):
            y_hat = self(x)
            loss_value = loss(y_hat, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

    def predict(self, x):
        return self(x)
            

