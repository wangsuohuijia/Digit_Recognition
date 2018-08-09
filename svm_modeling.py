'''
Reference:
http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py
http://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html#sphx-glr-auto-examples-svm-plot-rbf-parameters-py
'''

import os
import numpy as np
# import matplotlib.pyplot as plt
from time import time
from sklearn.svm import SVC
from sklearn import metrics
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from dir_info import model_dir, data_dir

# TODO - load mnist data
mnist_data = np.load(os.path.join(data_dir, 'mnist.npz'))
x_train = mnist_data['x_train']
y_train = mnist_data['y_train']
x_test = mnist_data['x_test']
y_test = mnist_data['y_test']

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# TODO - 创建多个SVM模型，然后运行

# C_range = [1]
# gamma_range = [0.01]
# C_range = np.logspace(-2, 10, 10)
# gamma_range = np.logspace(-9, 3, 10)
# param_grid = dict(gamma=gamma_range, C=C_range)
# cv = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)

# TODO - uniformization
x_train /= 255
x_test /= 255

# TODO - 定义模型
model = SVC()
# grid = GridSearchCV(SVC(probability=True),
#                     param_grid=param_grid,
#                     cv=cv,
#                     # n_jobs=4,
#                     verbose=True)

st = time()
# grid.fit(x_train, y_train)
model.fit(x_train, y_train)
et = time()
print('svm training time is %.5f sec' % (et-st))

# joblib.dump(grid, 'svm_grid.pkl')
joblib.dump(model, 'svm.pkl')

# TODO - 评估模型
# prediction = grid.predict(x_test)
prediction = model.predict(x_test)

print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_test, prediction)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

