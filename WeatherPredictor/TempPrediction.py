
"""
This function builds and trains a model in Tensorflow to predict the temperature 24 hours from the current temperature.
The input is hourly weather data from NOAA, consisting of temperature, realtive humidity, dewpoint, pressure.
The output is a set of probabilities for the temperature 24 hours from current input, as degrees from -40 to 120 degC
"""

import tensorflow as tf
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt


class tempModel(object):
    """
    Class used to build model utilizing LSTM and multilayer dense networks
    Input consists of [temp humidity dewpoint pressure] * number of stations
    """
    def __init__(self, isTraining = True, truncated_backprop = 7*24,
                       batch_size = 1000, number_of_stations = 3, state_size = 10):
        self.isTraining = isTraining
        self.truncated_backprop = truncated_backprop
        self.batch_size = batch_size
        self.inputLength = 4 * number_of_stations
        self.state_size = state_size
        if self.isTraining:
            self.batch_size = 1
        # Build model
        self._buildModel()

    def _buildModel(self):
        # Graph Place Holders / Variables
        with tf.name_scope('inputs'):
            input_batch_PH = tf.placeholder(tf.float32, [self.batch_size, 
                                                         self.truncated_backprop_num,
                                                         self.inputLength],
                                            name='input_batch')
        with tf.name_scope('targets'):
            target_batch_PH = tf.placeholder(tf.float32, [self.batch_size,
                                                          self.truncated_backprop_num,
                                                          1],
                                             name='target_batch')
            if self.isTraining:
                target_batch_cur = target_batch_PM[:,-24:,:] # Take only the last 24 hours of results for cost calc
                target_batch_cur = tf.one_hot(target_batch_cur)
            else:
                target_batch_cur = tf.one_hot(target_batch_cur) # Take everything when running predictions
            
            # LSTM / Dynamic RNN
            with tf.name_scope('LSTM'):
                cell = tf.contrib.rnn




