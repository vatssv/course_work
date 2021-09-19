import scipy.io
import numpy as np
import sys
import scipy.stats

data = scipy.io.loadmat('fashion_mnist.mat') 

class_0 = []
class_1 = []
data_features = []

for i in range(0, len(data['trX'])):
    feature_1 = np.mean(data['trX'][i], axis=0)
    feature_2 = np.std(data['trX'][i])
    data_features.append([feature_1, feature_2])
    if data['trY'][0][i] == 0:
        class_0.append([feature_1, feature_2])
    else:
        class_1.append([feature_1, feature_2])   

f_0_class_0 = [i[0] for i in class_0]
f_1_class_0 = [i[1] for i in class_0]
f_0_class_1 = [i[0] for i in class_1]
f_1_class_1 = [i[1] for i in class_1]

f_0_y_0_mu = np.mean(f_0_class_0, axis=0)
f_0_y_0_sigma = np.std(f_0_class_0)
f_1_y_0_mu = np.mean(f_1_class_0, axis=0)
f_1_y_0_sigma = np.std(f_1_class_0)

f_0_y_1_mu = np.mean(f_0_class_1, axis=0)
f_0_y_1_sigma = np.std(f_0_class_1)
f_1_y_1_mu = np.mean(f_1_class_1, axis=0)
f_1_y_1_sigma = np.std(f_1_class_1)

test_data = []

for i in range(0, len(data['tsX'])):
    feature_1 = np.mean(data['tsX'][i], axis=0)
    feature_2 = np.std(data['tsX'][i])
    test_data.append([feature_1, feature_2])

test_labels = [data['tsY'][0]]

test_pred = []

for i in range(0, len(test_data)):
  p_0_feature_1 = scipy.stats.norm.pdf(test_data[i][0], f_0_y_0_mu, f_0_y_0_sigma)
  p_0_feature_2 = scipy.stats.norm.pdf(test_data[i][1], f_1_y_0_mu, f_1_y_0_sigma)
  p_1_feature_1 = scipy.stats.norm.pdf(test_data[i][0], f_0_y_1_mu, f_0_y_1_sigma)
  p_1_feature_2 = scipy.stats.norm.pdf(test_data[i][1], f_1_y_1_mu, f_1_y_1_sigma)
  p_0 = p_0_feature_1 * p_0_feature_2 * 0.5
  p_1 = p_1_feature_1 * p_1_feature_2 * 0.5
  
  if p_0 > p_1:
    test_pred.append(0)
  else:
    test_pred.append(1)

# Logistic Regression!


# def sigmoid(A):
#     return 1/(1 + np.exp(-A))


# def get_weights(X, Y, n_iterations, lr):
    
#     w = np.zeros(X.shape[1])
        
#     for i in range(n_iterations):
#         A = np.dot(X,w)    
#         preds = sigmoid(A)    # Calculating sigmoid of wT.X  
        
#         gradient = np.dot(X.T,(Y-preds))    # Finding gradients using gradient ascent of log-likelihood
        
#         w = w+lr * gradient     # Updating weights
    
#     return w

def accuracy(ground_truth, predictions):
    correct = 0
    for i, x in enumerate(ground_truth):
        if x == predictions[i]:
            correct += 1

    return correct/2000


w = np.random.randn(1,3)
w = np.array(w, dtype=float)
w = w[0]
test_pred_log = []
print('w', w)
gradient = 0.01
test_data = []

for i in range(0, len(data['tsX'])):
    feature_1 = np.mean(data['tsX'][i], axis=0)
    feature_2 = np.std(data['tsX'][i])
    test_data.append([feature_1, feature_2])

x = np.array(data_features, dtype=float)
ones = np.ones((12000, 1))
x = np.hstack((ones, x))
for i in range(0, 400):
    t = np.dot(x, w)
    z = 1/(1 + np.exp(-t))
    y = data['trY'][0]
    w = w + gradient * (np.dot(y-z, x))

ones = np.ones((2000, 1))
test_data = np.hstack((ones, test_data))
t = np.dot(test_data, w)
z = 1/(1 + np.exp(-t))

for y in z:
    if y <= 0.5:
        test_pred_log.append(0)
    else:
        test_pred_log.append(1)

print('Accuracy: ', accuracy(test_labels[0], test_pred_log))