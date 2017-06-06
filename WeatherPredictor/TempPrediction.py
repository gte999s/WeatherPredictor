
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
    def __init__(self, num_of_classes,  
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



if __name__ == "__main__":
    LOGDIR = os.getcwd() + '/logs'
    print("Running")

    """
    Parameter Section
    """
    station_target = '53885'
    num_of_train_steps = int(1e6)
    min_temp = -30
    max_temp = 120
    num_of_classes = len(range(min_temp,max_temp)) # model temperatures for -30 F to 120 F
    num_of_rnn_layers = 4
    learning_rate = 1
    truncated_backprop = 24*7

    """
    END Parameter Section
    """
    
    # Get Training Data Set
    train_data = pd.read_hdf('trainingData.hdf') # load dataframe

    b=train_data.columns.to_series().str.startswith('HourlyPrecip') # nan seems to mean no rain, so replace with 0
    train_data.loc[:,b]=train_data.loc[:,b].fillna(0)

    train_data.fillna(train_data.mean(),inplace=True) # replace all other missing data with mean of feature
    train_data.fillna(0,inplace=True) # remove instances of weather statiosn with all nans
    stations = np.unique(np.array(train_data.columns.to_series().str.split('_').tolist())[:,1]) # get station WBAN out of column name (last part is digits)
    measurement_types = np.unique(np.array(train_data.columns.to_series().str.split('_').tolist())[:,0]) # get measurement out of column name (first part is measurement desc from NOAA

    # Get Target Data
    target_data = train_data.loc[:,'DryBulbFarenheit_' + station_target].values
    target_data = target_data - min_temp
    target_data = round(target_data)

    # Get Normalized Input Data
    for type in measurement_types:
        print('Normalizing Measurement: ', type)
        b = train_data.columns.to_series().str.startswith(type)
        maxVal = train_data.loc[:,b].max().max()
        minVal = train_data.loc[:,b].min().min()
        input_data = (train_data.values - minVal) / (maxVal - minVal)

    def getFeeder(truncated_backprop, batch_size):
        total_size = batch_size*truncated_backprop
        randIndexs = np.random.randint(24,input_data.shape[0]-25,total_size)
        
        if batch_size is 0:
            inputs = np.zeros((batch_size,truncated_backprop,input_data.shape[2]),dtype=tf.float32)
            targets = np.zeros((batch_size,truncated_backprop,1),dtype=tf.float32)
            for batch in np.arange(batch_size):
                selection = np.arange(randIndexs[batch],randIndexs[batch]+truncated_backprop)
                inputs[batch,:,:] = input_data[selection]
                targets[batch,:,1] = target_data[selection],
        else: 
            inputs = input_data[-truncated_backprop:]
            targets = input_data[-truncated_backprop:]

        feed = {
                    'input_batch_PH': inputs,
                    'target_batch_PH': targets
                }

        return feed
    
    # Build Training Model
    with tf.name_scope('Train'):
        with tf.variable_scope('Model', reuse=False):
            modelTrain = tempModel(num_of_classes=num_of_classes, isTraining=True, truncated_backprop=truncated_backprop,
                                   num_rnn_layers=num_rnn_layers, stations=stations)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_step = tf.train.AdadeltaOptimizer(learning_rate = learning_rate).minimize(loss=modelTrain.cost_mean, global_step=global_step)
            tf.summary.scalar('cost_mean',modelTrain.cost_mean)

    # Build Prediction Model
    with tf.name_scope('Pred'):
        with tf.variable_scope('Model', reuse=True):
            modelPred = tempModel(num_of_classes=num_of_classes, isTraining=False, truncated_backprop=truncated_backprop,
                                  num_rnn_layers=num_rnn_layers, stations=stations)
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

            feed = getFeeder(truncated_backprop, batch_size)   

            _summary, _train_step, _predictions = sess.run(
                [summary, train_step, modelTrain.temp_pred_prob],
                feed_dict=feed)            

            if step % 100 == 0:
                sv.summary_computed(sess, _summary)
                print("Step", step, 'Cost', modelTrain.cost_mean, 'of', num_of_train_steps)

        # Use trained model for predictions
        feed = getFeeder(truncated_backprop=24*30, batch_size=1)
      
        _predictions = sess.run([modelPred.temp_pred_prob], feed_dict=feed)
