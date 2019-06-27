import numpy as np 
import pandas as pd 
import scipy.sparse as scp 
import random

test_mask = np.random.binomial(1, 0.8, size=10*8954).reshape(10, 8954)


R = np.asarray([[int(1 + random.random()*3) for j in range(8954)] for i in range(10)])

R = np.multiply(R,test_mask)

R_sparse = scp.csc_matrix(R)

scp.save_npz('Bio_Dataset_test/R.npz', R_sparse)