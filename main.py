import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# data and starting coefficient
data = pd.read_csv('data/data-logistic.csv', header=None)
w1 = 0
w2 = 0
k = 0.1
l = data.shape[0]
C = 10
# slicing data
y = data.iloc[:, 0].to_numpy()
X = data.iloc[:, 1:3].to_numpy()
X1 = data.iloc[:, 1].to_numpy()
X2 = data.iloc[:, 2].to_numpy()

# main part, gradient descent
# general linear regression
for i in range(10000):
    w1_next = w1 + k * np.mean(y * X1 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2)))))
    w2_next = w2 + k * np.mean(y * X2 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2)))))
    if i != 0 and np.sqrt(np.power(w1_next - w1, 2) + np.power(w2_next - w2, 2)) <= (10 ** (-5)):
        print(i)
        break
    w1 = w1_next
    w2 = w2_next
W_without = [w1, w2]

w1, w2 = 0, 0
# linear regression L2-regularized
for i in range(10000):
    w1_next = w1 + k * np.mean(y * X1 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2))))) - k * C * w1
    w2_next = w2 + k * np.mean(y * X2 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2))))) - k * C * w2
    if i != 0 and np.sqrt(np.power(w1_next - w1, 2) + np.power(w2_next - w2, 2)) <= (10 ** (-5)):
        print(i)
        break
    w1 = w1_next
    w2 = w2_next
W_with = [w1, w2]

# finishing
task = open('answers/task1.txt', 'w')
sigmoid = 1 / (1 + np.exp(-W_without[0] * X1 - W_without[1] * X2))
task.write(str(round(roc_auc_score(y, sigmoid), 3)) + ' ')
sigmoid = 1 / (1 + np.exp(-W_with[0] * X1 - W_with[1] * X2))
task.write(str(round(roc_auc_score(y, sigmoid), 3)))
task.close()

print('❤️🏅🍺👨🏿‍💻')

# for testing
"""print(W_without)
print(W_with)
k = 0.15
w1_start = 0
w2_start = 0
w1, w2 = w1_start, w2_start
for i in range(10000):
    w1_next = w1 + k * np.mean(y * X1 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2)))))
    w2_next = w2 + k * np.mean(y * X2 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2)))))
    if i != 0 and np.sqrt(np.power(w1_next - w1, 2) + np.power(w2_next - w2, 2)) <= (10 ** (-5)):
        print(i)
        break
    w1 = w1_next
    w2 = w2_next

W_without = [w1, w2]
w1, w2 = w1_start, w2_start
for i in range(10000):
    w1_next = w1 + k * np.mean(y * X1 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2))))) - k * C * w1
    w2_next = w2 + k * np.mean(y * X2 * (1 - 1/(1 + np.exp(-y * (w1 * X1 + w2 * X2))))) - k * C * w2
    if i != 0 and np.sqrt(np.power(w1_next - w1, 2) + np.power(w2_next - w2, 2)) <= (10 ** (-5)):
        print(i)
        break
    w1 = w1_next
    w2 = w2_next
W_with = [w1, w2]

print(W_without)
print(W_with)

sigmoid = 1 / (1 + np.exp(-W_without[0] * X1 - W_without[1] * X2))
print(str(round(roc_auc_score(y, sigmoid), 3)) + ' ')
sigmoid = 1 / (1 + np.exp(-W_with[0] * X1 - W_with[1] * X2))
print(str(round(roc_auc_score(y, sigmoid), 3)))"""

