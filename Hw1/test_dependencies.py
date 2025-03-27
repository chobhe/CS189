import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")
import numpy as np
import matplotlib.pyplot as plt
import os 
from sklearn import svm
from scipy import io
import os
print("Current working directory:", os.getcwd())
for data_name in ["mnist", "spam", "cifar10"]:
    if not os.path.exists("data/%s_data.mat" % data_name):
        raise FileNotFoundError(f"Data file '{'data/%s_data.mat' % data_name}' not found!")
    data = io.loadmat("data/%s_data.mat" % data_name)
print("\nloaded %s data!" % data_name)
fields = "test_data", "training_data", "training_labels"
for field in fields:
    print(field, data[field].shape)