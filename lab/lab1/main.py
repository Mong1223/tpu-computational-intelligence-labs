
import numpy as np
import random
import pandas as pd

from MyLoadData import df
from sklearn.model_selection import train_test_split

class NeuralNetwork():
    def __init__(self, ):
        self.inputSize = 8
        self.outputSize = 1
        self.hiddenSize = 8

        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)

        self.error_list = []
        self.limit = 0.5
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0

    def forward(self, X):
        self.z = np.matmul(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.matmul(self.z2, self.W2)
        o = self.sigmoid(self.z3)
        return o

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = np.matmul(self.o_delta,
                                  np.matrix.transpose(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += np.matmul(np.matrix.transpose(X), self.z2_delta)
        self.W2 += np.matmul(np.matrix.transpose(self.z2),
                             self.o_delta)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            o = self.forward(X)
            self.backward(X, y, o)
            self.error_list.append(np.abs(self.o_error).mean())

    def predict(self, x_predicted):
        return self.forward(x_predicted).item()

    def score(self, predict, y):
        cnt = np.sum(predict == y)
        return (cnt / len(y)) * 100

    # def view_error_development(self):
    #     plt.plot(range(len(self.error_list)), self.error_list)
    #     plt.title('Mean Sum Squared Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')

    def test_evaluation(self, input_test, output_test):
        for i, test_element in enumerate(input_test):
            if self.predict(test_element) > self.limit and \
                    output_test[i] == 1:
                self.true_positives += 1
            if self.predict(test_element) < self.limit and \
                    output_test[i] == 1:
                self.false_negatives += 1
            if self.predict(test_element) > self.limit and \
                    output_test[i] == 0:
                self.false_positives += 1
            if self.predict(test_element) < self.limit and \
                    output_test[i] == 0:
                self.true_negatives += 1
        print('True positives: ', self.true_positives,
              '\nTrue negatives: ', self.true_negatives,
              '\nFalse positives: ', self.false_positives,
              '\nFalse negatives: ', self.false_negatives,
              '\nAccuracy: ',
              (self.true_positives + self.true_negatives) /
              (self.true_positives + self.true_negatives +
               self.false_positives + self.false_negatives))





if __name__ == '__main__':
    print('sssss')
    # df = pd.read_csv("diabetes.csv")
    df_train, df_test = train_test_split(df, test_size=0.3)

    x_train = df_train.values[:, :8]
    y_train = df_train.values[:, 8]
    x_test = df_test.values[:, :8]
    y_test = df_test.values[:, 8]

    clr = NeuralNetwork()

    clr.train(x_train, y_train, epochs=50000)
    pre_y = clr.predict(x_test) # predict

    score = clr.score(pre_y, y_test)
    print('res', pre_y)
    print('y', y_test)
    print(score)


