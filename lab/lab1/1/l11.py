import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random


class my_NN(object):
    def __init__(self):
        self.input = 13  # feature numbers
        self.output = 1  # class number
        self.hidden_units = 13  # single layer

        # initialize matrix of weights;
        # np.random.seed(1)
        # weight1: input -> hidden layer
        self.w1 = np.random.randn(self.hidden_units, self.input) * 0.01  # 8*8 matrix
        self.b1 = np.zeros((self.hidden_units, 1))
        # weight2: hidden layer -> output
        self.w2 = np.random.randn(self.output, self.hidden_units) * 0.01  # 8*1 matrix
        self.b2 = np.zeros((self.output, 1))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_prime(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _forward_propagation(self, X):
        self.z2 = np.dot(self.w1, X) + self.b1
        self.a2 = self._sigmoid(self.z2)
        self.z3 = np.dot(self.w2, self.a2) + self.b2
        self.a3 = self._sigmoid(self.z3)
        return self.a3

    def _loss(self, predict, y):
        m = y.shape[1]
        epsilon = 1e-9
        logprobs = np.sum(np.multiply(y, np.log(predict)) + np.multiply((1 - y), np.log(1 - predict)))
        #
        # logprobs = np.average(np.multiply(y,np.log(predict + epsilon)) + np.multiply((1 - y), np.log(1 - predict + epsilon)))
        loss = - logprobs / m
        return loss

    def _backward_propagation(self, X, y):
        predict = self._forward_propagation(X)
        m = X.shape[1]
        delta3 = predict - y

        self.dw2 = (1 / m) * np.dot(delta3, self.a2.T)
        self.db2 = (1 / m) * np.sum(delta3, axis=1, keepdims=True)

        delta2 = np.multiply(np.dot(self.w2.T, delta3), self._sigmoid_prime(self.z2))
        self.dw1 = (1 / m) * np.dot(delta2, X.T)
        self.db1 = (1 / m) * np.sum(delta2, axis=1, keepdims=True)

    def _update(self, learning_rate=.1):
        self.w1 = self.w1 - learning_rate * self.dw1
        self.b1 = self.b1 - learning_rate * self.db1
        self.w2 = self.w2 - learning_rate * self.dw2
        self.b2 = self.b2 - learning_rate * self.db2

    def train(self, X, y, iteration=10000):
        for i in range(iteration):
            y_hat = self._forward_propagation(X)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X, y)
            self._update()
            if i % 100 == 0:
                print("loss: ", loss)

    def predict(self, X):
        y_hat = self._forward_propagation(X)
        y_hat = [1. if i[0] >= 0.5 else 0. for i in y_hat.T]
        return np.array(y_hat)

    def score(self, predict, y):
        print(predict, ' : ', y[0])
        cnt = np.sum(predict == y[0])
        return (cnt / y.shape[1]) * 100


def get_samples(x, y, shp, num_k=1, num_in_sample=500):
    # x = pd.DataFrame(x)
    # y = pd.DataFrame(y)

    samples_x = []
    samples_y = []
    for i in range(num_k):
        sample_x = np.zeros(shape=(num_in_sample, 13))
        sample_y = np.zeros(shape=(num_in_sample, 1))
        for j in range(num_in_sample):
            if j == num_in_sample:
                pass
            random_index = random.randrange(int(shp))
            sample_x[j] = x[random_index]
            sample_y[j] = y[random_index]

        # sample_x=np.ndarray(sample_x)
        # sample_y=np.ndarray(sample_y)
        samples_x.append(sample_x)
        samples_y.append(sample_y)
    return samples_x, samples_y


if __name__ == '__main__':
    df = pd.read_csv('heart.csv', header=None)
    df = df.drop(0)
    df[13] = df[13].astype(np.int64)

    # We separate the two class:
    # df_d0 = df[df[13] == 0]
    # df_d1 = df[df[13] == 1]

    # and random sample the same amount of instances from the higher represented '0' class to match the available '1' class instances
    # df_d0_samp = df_d0.sample(268, replace=False)
    # df = pd.concat([df_d1, df_d0_samp]) #putting back the two class together
    # print(df)

    # separate the independent variables from the target variable
    data = df.loc[:, 0:12]
    target = df[[13]]

    data = StandardScaler().fit_transform(data)

    # divide the data into a training set and a test set

    x_train, x_test, y_train, y_test = train_test_split(data, target[13], test_size=0.3, random_state=42)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]).astype('float32').T
    y_train = np.array(y_train).reshape(np.array(y_train).shape[0], 1)
    y_test = np.array(y_test).reshape(np.array(y_test).shape[0], 1).T

    num_k = 5
    shp = x_train.shape[0]
    num_in_sample = 700
    features = 13
    results = []
    for k in range(1, num_k + 1, 2):
        num = k
        # Выборки
        x_traink, y_traink = get_samples(x_train, y_train, shp, num, num_in_sample)

        Parameters_f = []

        for i in range(num):
            clr = my_NN()
            df_X = pd.DataFrame(x_traink[i].T)
            df_Y = pd.DataFrame(y_traink[i].T)

            x_trainf = df_X.values
            y_trainf = df_Y.values
            clr.train(x_trainf, y_trainf, iteration=50000)
            pre_y = clr.predict(x_test)  # predict

            Parameters_f.append(pre_y)

        res = 0
        for i in range(num):
            res = res + Parameters_f[i]
        result = []
        for i in res:
            if i >= (num / 2):
                result.append(1)
            else:
                result.append(0)
        clr = my_NN()
        score = clr.score(result, y_test)
        # print('res', result)
        # print('y', Y_test)
        print(score)
        results.append([k, score])

        print('hi')
    print(results)

    # clr = my_NN()
    # clr.train(x_train, y_train, iteration=50000)
    # pre_y = clr.predict(x_test)  # predict
    # score = clr.score(pre_y, y_test)
    # print('score = ',score)
