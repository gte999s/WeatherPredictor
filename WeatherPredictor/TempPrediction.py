
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
    def __init__(self, num_of_classes, input_data, target_data,
                       truncated_backprop = 7*24,  isTraining = True, 
                       batch_size = 1000, stations = [3001, 3002, 3003], state_size = 10,
                       dropout_keep_prob = .9, num_rnn_layers = 10):
        self.isTraining = isTraining
        self.truncated_backprop = truncated_backprop
        self.batch_size = batch_size
        self.inputLength = 4 * len(stations)
        self.state_size = state_size
        self.dropout_keep_prob = dropout_keep_prob
        self.num_rnn_layers = num_rnn_layers
        self.num_of_classes = num_of_classes
        self.stations = stations
        self.input_data = input_data
        self.target_data = target_data

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
                target_batch_cur = tf.one_hot(target_batch_cur, depth=self.num_of_classes)
            else:
                target_batch_cur = tf.one_hot(target_batch_cur, depth=self.num_of_classes) # Take everything when running predictions
            
            # LSTM / Dynamic RNN
        with tf.name_scope('LSTM'):
            doReuse = not self.isTraining
            cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True, reuse=doReuse)
            if self.isTraining:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.num_rnn_layers, state_is_tuple=True)
            outputs_batch, _ = tf.nn.dynamic_rnn(cell, input_batch_PH, dtype=tf.float32)
            outputs = tf.reshape(outputs_batch,[-1, self.state_size])

        with tf.name_scope('LinearLayer'):
            W = tf.get_variable('W',shape=(self.state_size, self.num_of_classes),dtype=tf.float32)
            b = tf.get_variable('b',shape=(1,self.num_of_classes))
            temp_pred_logits = tf.matmul(outputs, W) + b
            self.temp_pred_prob = tf.sparse_softmax(temp_pred_logits)

        if self.isTraining:

            with tf.name_scope('cost_calc'):
                costs = tf.nn.sparse_softmax_cross_entropy_with_logits(temp_pred_logits,target_batch_cur)
                self.cost_mean = tf.reduce_mean(costs)

    def generateFeeder():
        ## Generates an inputs and target feed for tensorflow 
        
    


        return feed, feed_length


if __name__ == "__main__":
    LOGDIR = os.getcwd() + '/logs'
    print("Running")

    # Parameters
    num_of_train_steps = int(1e6)
    min_temp = -30
    max_temp = 120
    num_of_classes = len(range(min_temp,max_temp)) # model temperatures for -30 F to 120 F
    num_of_rnn_layers = 4
    stations = [3011, 3012, 3013]
    learning_rate = 1
    input_data = pd.read_hdf('trainingData.hdf')

    # Get Training Data
   
    
    # Build Training Model
    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=False):
            modelTrain = tempModel(num_of_classes=num_of_classes, isTraining=True, num_rnn_layers=num_rnn_layers, stations=stations)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_step = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss=modelTrain.cost_mean, global_step=global_step)
            tf.summary.scalar('cost_mean',modelTrain.cost_mean)

    # Build Prediction Model
    with tf.name_scope('Pred'):
        with tf.variable_scope('Model', reuse=True):
            modelPred = tempModel(num_of_classes=num_of_classes, isTraining=False, num_rnn_layers=num_rnn_layers, stations=stations)
            tf.summary.histogram('temp_pred_prob', modelPred.temp_pred_prob)

    # Merge all summaries
    summary = tf.summary.merge_all

    sv = tf.train.Supervisor(logdir=LOGDIR,
                             save_model_secs=10,
                             summary_op=None,
                             global_step=global_step)

    with sv.managed_session() as sess:

        # Do Training Steps
        for step in range(num_of_train_steps):
            if sv.should_stop():
                break

            feed = modelTrain.getFeeder()    

            _summary, _train_step, _predictions = sess.run(
                [summary, train_step, modelTrain.temp_pred_prob],
                feed_dict=feed)            

            if step % 100 == 0:
                sv.summary_computed(sess, _summary)
                print("Step", step, 'Cost', modelTrain.cost_mean, 'of', num_of_train_steps)

        # Use trained model for predictions
        feed, feed_length = modelPred.getFeeder()
      
        _predictions = sess.run([modelPred.temp_pred_prob], feed_dict=feed)
