import simpleml
import time
import pandas as pd
import numpy as np

model = simpleml.Log_regression()
model.load_data('test_data/x_train_cls.csv', 'test_data/t_train_cls.csv', 10)

time_start = time.time_ns()
model.train(50)
time_end = time.time_ns()
reg_time = time_end - time_start
print("Duration:", reg_time / 1e9, "seconds")

test_x = pd.read_csv('test_data/x_test_cls.csv', header=None).to_numpy().flatten()
test_y = pd.read_csv('test_data/t_test_cls.csv', header=None).to_numpy().flatten()
pred = np.array(model.test(test_x))
print("Test acc:", np.sum(test_y == pred) / len(test_y))
