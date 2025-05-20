import torch
import numpy as np
import pickle
from random import shuffle
from model import *
from datautils import prepare_data

torch.set_default_dtype(torch.float64)

with open('final_array.pkl', 'rb') as f:
    train_data, train_label, val_data, val_label = pickle.load(f)
train_data = torch.tensor(train_data, requires_grad=False)
train_label = torch.tensor(train_label, requires_grad=False)
val_data = torch.tensor(val_data, requires_grad=False)
val_label = torch.tensor(val_label, requires_grad=False)

# train_data, train_label, val_data, val_label = prepare_data('sag_healthy.csv')

acc, train_time = nntli_train(train_data, train_label, val_data, val_label,'naval')
# acc, train_time = tlnn_train_time_weight(train_data, train_label, val_data, val_label)

a = 1

