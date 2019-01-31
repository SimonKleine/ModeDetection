import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


test = [1,2,3,4,5]
k = [0, 1, 2]
test1 = test[k]

print(test)
print(test1)