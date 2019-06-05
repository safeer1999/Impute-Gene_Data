#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:27:44 2019

@author: qcri-inter9
"""

import pandas as pd
import numpy as np
import scipy.sparse as scp
import tensorflow as tf
import tensorflow_probability as tfp
import sys

R = np.load("./Experiment1/BioDataset1_output/R_norm_complete_matrix.npy")
recons = np.load("./Experiment1/BioDataset1_output/recons_complete_matrix.npy")

r_batch = R[:,:170][:100,:]
recons_batch = recons[:,:170][:100,:]

correlation = tf.abs(tfp.stats.correlation(r_batch,recons_batch, sample_axis = 0, event_axis = None))
mean = tf.reduce_mean(correlation)

nan_indices = tf.is_nan(correlation)

correlation_excludeNan = tf.where(nan_indices, tf.constant(1, dtype = "float64", shape = [170]), correlation)

mean = tf.reduce_mean(correlation_excludeNan)

print_correl = tf.print(correlation, output_stream = sys.stdout)
print_mean = tf.print(mean, output_stream = sys.stdout)
print_nan_indices = tf.print(nan_indices, output_stream = sys.stdout)
print_correlation_excludeNan = tf.print(correlation_excludeNan, output_stream = sys.stdout)




with tf.Session() as sess :
    sess.run([print_correlation_excludeNan, print_mean])


    



