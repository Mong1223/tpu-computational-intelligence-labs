import numpy as np
import random
import pandas as pd

from MyLoadData import df


class my_NN(object):
    def __init__(self):
        self.input = 8  # feature numbers
        self.output = 1  # class number
        self.hidden_units = 8  # single layer

        # initialize matrix of weights;
        # np.random.seed(1)
        # weight1: input -> hidden layer
        self.w1 = np.random.randn(self.input, self.hidden_units)  # 8*8 matrix
        # self.b1 = np.zeros((self.hidden_units, 1))
        # weight2: hidden layer -> output
        self.w2 = np.random.randn(self.hidden_units, self.output) # 8*1 matrix
        # self.b2 = np.zeros((self.output, 1))


    def _forward_propagation(self, X):
        # self.z2 = np.dot(self.w1.T, X.T) + self.b1
        self.z2 = np.dot(self.w1.T, X.T)
        self.a2 = self._sigmoid(self.z2)
        # self.z3 = np.dot(self.w2.T, self.a2) + self.b2
        self.z3 = np.dot(self.w2.T, self.a2)
        self.a3 = self._sigmoid(self.z3)
        return self.a3

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))


    def _loss(self, predict, y):
        m = y.shape[0]
        # logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        epsilon = 1e-5
        logprobs = np.multiply(y, np.log(predict + epsilon)) + np.multiply((1 - y), np.log(1 - predict + epsilon))
        loss = - np.sum(logprobs) / m
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        m = X.shape[0]
        delta3 = predict - y
        # dz3 = np.multiply(delta3, self._sigmoid_prime(self.z3))
        # self.dw2 = (1/m) * np.sum(delta3*self.a2)
        self.dw2 = (1 / m) * np.sum(np.multiply(delta3, self.a2), axis=1).reshape(self.w2.shape)
        # self.db2 = (1 / m)*np.sum(delta3, axis=1, keepdims=True)
        # self.db1 = (1 / m)*delta3
        # delta2 = delta3 * self.w2 * self._sigmoid_prime(self.z2)
        delta2 = delta3 * self.w2 * self._sigmoid_prime(self.z2)
        self.dw1 = (1 / m) * np.dot(X.T, delta2.T)
        # self.db1 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)
        # self.db2 = (1 / m)*delta2

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _update(self, learning_rate=0.1):
        self.w1 = self.w1 - learning_rate * self.dw1
        # self.b1 = self.b1 - learning_rate * self.db1
        self.w2 = self.w2 - learning_rate * self.dw2
        # self.b2 = self.b2 - learning_rate * self.db2

    def train(self, X, y, iteration=100000):
        for i in range(iteration):
            y_hat = self._forward_propagation(X)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X, y)
            self._update()
            if i % 1000 == 0:
                print("loss: ", loss)

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        y_hat = [1. if i[0] >= 0.5 else 0. for i in y_hat.T]
        return np.array(y_hat)

    def score(self, predict, y):
        cnt = np.sum(predict == y)
        return (cnt / len(y)) * 100


from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df_train, df_test = train_test_split(df, test_size=0.3)

    x_train = df_train.values[:, :8]
    y_train = df_train.values[:, 9]
    x_test = df_test.values[:, :8]
    y_test = df_test.values[:, 9]

    clr = my_NN()

    clr.train(x_train, y_train, iteration=100000)
    pre_y = clr.predict(x_test) # predict

    score = clr.score(pre_y, y_test)
    print('res', pre_y)
    print('y', y_test)
    print(score)


