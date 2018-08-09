import os

data_dir = './data'
model_dir = './model'
result_dir = './results'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
