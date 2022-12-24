import simpleml
import time
import pandas as pd
import numpy as np
import unittest

class TestModelMethods(unittest.TestCase):
    # regression
    def test_reg_train(self):
        # weight from simpleML
        model = simpleml.Regression()
        model.load_data('test_data/x.csv', 'test_data/y_bios.csv')
        model.train()
        weight_ml = np.array(model.weight)

        # weight from pure python
        X = np.array(pd.read_csv('test_data/x.csv', header=None))
        y = np.array(pd.read_csv('test_data/y_bios.csv', header=None)).flatten()
        phi_train_x = np.concatenate((np.ones((len(X), 1)), X), axis=1)
        weight = np.dot(np.dot(np.linalg.inv(np.dot(phi_train_x.T, phi_train_x)), phi_train_x.T), y)
        self.assertTrue(np.min(np.isclose(weight_ml, weight)))
    

    # logistic regression
    def test_logreg_train(self):
        # weight from simpleML
        model = simpleml.Log_regression()
        model.load_data('test_data/x_train_cls.csv', 'test_data/t_train_cls.csv', 10)
        model.train(50)
        weight_ml = np.array(model.weight).reshape((10, 784))

        # weight from pure python
        train_x = pd.read_csv('test_data/x_train_cls.csv', header=None).to_numpy()
        n_cls = 10
        train_label = np.zeros([n_cls, 1])
        for i in range(n_cls):
            label_temp = np.zeros([n_cls, 1])
            label_temp[i, 0] = 1
            label_temp = np.repeat(label_temp, 128, axis=1)
            train_label = np.hstack((train_label, label_temp[:, 32:]))
        train_label = train_label[:, 1:]

        epoch = 50
        lr = 0.0005

        weight = np.zeros([n_cls, 784])
        for iter in range(epoch):
            train_a = np.dot(weight, train_x.T)
            train_y = np.exp(train_a) / np.sum(np.exp(train_a), axis=0)
            weight -= lr * np.dot((train_y - train_label), train_x)
        
        self.assertTrue(np.min(np.isclose(weight_ml, weight)))

if __name__ == '__main__':
    unittest.main()
