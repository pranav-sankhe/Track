import tensorflow as tf  
import numpy as np 
import os 
import sys
sys.path.insert(0, '../data')
import utils
import pandas as pd
import random
import hparams

test_flag = 0
data = pd.read_csv(hparams.data_path)

# # Graph weights
# weights = {
#     'hidden': tf.Variable(tf.random_normal([hparams.num_nodes, hparams.lstm_units])), # Hidden layer weights
#     # 'BN':     tf.Variable(tf.random_normal([n_input, n_hidden])),  #BatchNormalization weights
#     # 'full_connected': tf.Variable(tf.random_normal([n_input, n_hidden])),
#     'out': tf.Variable(tf.random_normal([hparams.lstm_units, hparams.num_labels], mean=1.0))
# }
# biases = {
#     'hidden': tf.Variable(tf.random_normal([hparams.lstm_units])),
#     'out': tf.Variable(tf.random_normal([hparams.num_labels]))
# }


def build_lstm_layers(lstm_sizes, inputs, keep_prob_, batch_size):

    lstms = [tf.contrib.rnn.LayerNormBasicLSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
    return lstm_outputs, final_state


def apply_dense_layer(inputs):
    logits = tf.layers.dense(inputs, hparams.dense_units)
    return logits

def create_labels(num_labels,dist):
    labels = np.histogram(dist, bins=num_labels-1)[1]
    return labels   

def label_to_dist(label, label_vals):
    distance = label_vals[label]
    return distance

def get_label(val, label_vals):
    val = [val]*len(label_vals)
    vals = np.array(val)
    diff = np.subtract(vals, label_vals)  
    diff = np.abs(diff)
    return np.argmin(diff)


def core_model(lstm_sizes, inputs, keep_prob_, batch_size):
    outputs, state = build_lstm_layers(lstm_sizes, inputs, keep_prob_, batch_size)
    outputs = outputs[:,-1,:]
    prev_layer = outputs
    fc_size = hparams.fc_size
    
    next_layer = []
    for l in range(len(fc_size)):    
        next_layer = utils.fully_connected(fc_size[l], prev_layer, 'FC_' + str(l))
        prev_layer = next_layer

    logits = next_layer
    # logits = apply_dense_layer(outputs)
    
    
    return logits


def compute_loss(gt_labels, logits):    
    
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=gt_labels, logits=logits)
    # target_weights = tf.sequence_mask(
    #     hparams.batch_size, max_time, dtype=logits.dtype)
    # if hparams.TIME_MAJOR:
    #     target_weights = tf.transpose(target_weights)

    # loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
    return crossent	

def compute_loss_regression(grnd_truth, predictions):

    reg_loss = tf.losses.mean_squared_error(
    labels = grnd_truth,
    predictions = predictions)
    # weights=1.0,
    # scope=None,
    # loss_collection=tf.GraphKeys.LOSSES,
    # reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
    reg_loss = reg_loss*3968
    return reg_loss

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm



def train():

    
    input_rssi = tf.placeholder(tf.float32, [None, hparams.num_time, hparams.num_nodes])
    # input_test_rssi = tf.placeholder(tf.float32, [hparams.test_labels_length, hparams.num_time, hparams.num_nodes])
    distance  = tf.placeholder(tf.float32, [hparams.batch_size,1])
    
    # correct_answer = tf.placeholder(tf.int64, [hparams.batch_size,1])
    # test_bool = tf.constant(0)
    # batchsz = tf.placeholder(tf.int32)

    prediction = core_model(hparams.lstm_sizes, input_rssi, hparams.keep_prob_, hparams.batch_size)
    # logits_test = core_model(hparams.lstm_sizes, input_rssi, hparams.keep_prob_, hparams.test_labels_length)

    # logits_test = core_model(hparams.lstm_sizes, input_test_rssi, hparams.keep_prob_, hparams.test_labels_length)
    # logits2 = core_model(hparams.lstm_sizes, input_test_rssi, hparams.keep_prob_, hparams.batch_size)
    #2 logits giving error
    # Compute loss and gradients, but don't apply them yet
    #batch_size = input_vids_list[i].get_shape().as_list()[0]
    
    # prediction = prediction*tf.sqrt(3928.4198316935376) + 179.96025187907708
    loss = compute_loss_regression(distance, prediction)
    #loss = tf.reduce_mean(loss)
    # test time accuracy calculation
    # prediction = tf.argmax(logits, 1)
    # # prediction_test = tf.argmax(logits_test,2)

    # equality = tf.equal(prediction, correct_answer)
    # accuracy = np.sum(tf.cast(equality, tf.float32))/hparams.batch_size
    
    # tf.summary.scalar('accuracy', tf.reduce_mean(accuracy))
    

    with tf.name_scope("compute_gradients"):
        # compute_gradients` returns a list of (gradient, variable) pairs
        params = tf.trainable_variables()

        for var in params:
            tf.summary.histogram(var.name, var)
        
        grads = tf.gradients(xs=params, ys=loss, colocate_gradients_with_ops=True)    # optimizer.compute_gradients(loss)
        clipped_grads = grads
        #clipped_grads, grad_norm_summary, grad_norm = gradient_clip(grads, max_gradient_norm=hparams.max_gradient_norm)
        grad_and_vars = zip(clipped_grads, params)

    
    # lr = learning_rate = tf.constant(hparams.LR)
    # # warm-up
    # learning_rate = _get_learning_rate_warmup(global_step, lr)
    # # decay
    # learning_rate = _get_learning_rate_decay(hparams)
    
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(hparams.LR)
    apply_gradient_op = optimizer.apply_gradients(grad_and_vars, global_step)
    
    tf.summary.scalar('loss', loss)
    
    saver = tf.train.Saver()
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        )
    session_config.gpu_options.allow_growth = True
    
    sess = tf.InteractiveSession(config=session_config)

    initializer = tf.contrib.layers.xavier_initializer()
    
    with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        # with tf.variable_scope(tf.get_variable_scope(), initializer=initializer):
        init = tf.global_variables_initializer()
        sess.run(init)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(hparams.log_dir ,sess.graph)
        global test_flag
        #saver = tf.train.Saver()
        for j in range(hparams.num_iterations): 
            
            print "Training:: iteration: ", j 
            test_flag = 0
            inputs = {"Datafile":  hparams.data_path}           # Get the data filepath 

            dist = utils.get_dist_data(inputs)                  # Get array containing distance of the object node to router 
            dist = utils.normalize_dist(dist) 
            X = utils.get_rssi_data(inputs)                     # Get RSSI data. Shape =  num_data_points * num_nodes
            # X = utils.get_eta_data(inputs)
            X = utils.normalize_data(X)
            

            # label_vals = create_labels(hparams.num_labels,dist)
            # dist_labels = []
            # for dist_val in dist: 
            #     dist_labels.append(get_label(dist_val, label_vals))
            dist_labels = dist

            # temp = np.column_stack((dist_labels, X))                           # create a tenporary ensemble of features and target variable
            # np.random.shuffle(temp)                                # Random shuffle    
            # dist_labels = temp[:,0]                                # seperate feartures and variables     
            # X = temp[:,1:]

            length = len(dist_labels)                                  #Get length of the dataset     
            train_length = int(length*0.8)                      # Train dataset length 
            test_length = int(length*0.2)                       # Test dataset length 
            
            X_train = X[0:train_length]                         # Train dataset (features)
            X_test = X[-test_length:]                           # Test dataset (features)
                
            dist_train = dist_labels[0:train_length]                   # Train dataset (labels)
            dist_test = dist_labels[-test_length:]                     # Test dataset (labels)    
            

            epochs = (len(dist_train) - 4)/hparams.batch_size         # number of epochs for training 
            
            avg_cost = 0.0

            for i in range(epochs):
                print "Training:: Epoch ", i 
                
                train_batch_rssi = utils.getbatch_X(i, X_train)#[i:i+hparams.batch_size]          # get batched data (features)
                train_batch_labels = utils.getbatch_labels(i, dist_train)     # get batched data (labels)
                train_batch_labels = np.reshape(train_batch_labels,(hparams.batch_size,1))

                _, loss_val, pred, summary = sess.run(
                    [apply_gradient_op, loss, prediction, merged],
                    feed_dict={
                        input_rssi: train_batch_rssi, 
                        distance: train_batch_labels,
                        # batchsz: hparams.batch_size
                    }
                ) 
                train_writer.add_summary(summary)
                # print pred*np.sqrt(1315.2341209347906) + 198.05518950451153
                avg_cost += loss_val / epochs
                print "loss: ", loss_val
            print "Average Cost: ", avg_cost
            save_path = saver.save(sess, hparams.save_path)
            print "Model saved"    
            # total_iter_test_error = 0

        
            # # import pdb; pdb.set_trace();
            # test_epochs = test_length/(hparams.batch_size*hparams.num_time)
            # for k in range(test_epochs - 1):
            #     test_flag = 1    
            #     test_rssi = utils.getbatch_X(k,X_test)
            #     test_labels = utils.getbatch_labels(k,dist_test)
            #     test_labels = np.reshape(test_labels,(hparams.batch_size,1))
            #     #tf.summary.scalar('accuracy', tf.reduce_sum(loss))
            #     accuracy_val, pred_val = sess.run(
            #             [loss, prediction],
            #             feed_dict={
            #                 input_rssi: test_rssi, 
            #                 distance: test_labels,
            #                 test_bool: 1
            #                 # batchsz: hparams.test_labels_length
            #             }
            #         )
            #     #pred_val = pred_val*np.sqrt(1315.2341209347906) + 198.05518950451153
            #     #test_labels = test_labels#*np.sqrt(1315.2341209347906) + 198.05518950451153
            #     temp = np.stack((pred_val, test_labels), axis=-1)

            #     print temp
            #     # temp = temp[:,0,:]

            #     # if k==0: 
            #     #     answer = temp
            #     # else:
                    
            #     #     answer = np.vstack((answer, temp))

            #     # import pdb;pdb.set_trace()    
            #     accuracy_val = np.sum(accuracy_val)
            #     total_iter_test_error = accuracy_val + total_iter_test_error
            #     accuracy_list.append(total_iter_test_error)
            #     # np.save('accuracy', accuracy_list)

            # # answer = pd.DataFrame(answer)
            # # answer.to_csv('predictions.csv')
            # print "Testing Error ", total_iter_test_error/test_epochs

def test():
    input_rssi = tf.placeholder(tf.float32, [None, hparams.num_time, hparams.num_nodes])
    distance  = tf.placeholder(tf.float32, [hparams.batch_size,1])
    optimizer = tf.train.AdamOptimizer(hparams.LR)

    prediction = core_model(hparams.lstm_sizes, input_rssi, hparams.keep_prob_, hparams.batch_size)
    # prediction = prediction*np.sqrt(1315.2341209347906) + 198.05518950451153
    # loss = compute_loss_regression(distance, prediction)
    temp_pred = prediction*tf.sqrt(3928.4198316935376) + 179.96025187907708
    temp_distance = tf.to_float(distance)*tf.sqrt(1489.9552173615964) + 176.15052954295948#.15052954295948
    #temp_pred = tf.add(tf.multiply(prediction, tf.sqrt(1489)), 176)
    #temp_distance = tf.add(tf.multiply(distance, tf.sqrt(1490)), 176)
    accuracy = temp_pred - temp_distance #tf.metrics.mean_absolute_error(temp_pred, temp_distance)
    saver = tf.train.Saver()
    tf.summary.scalar('accuracy', tf.reduce_sum(accuracy))
    with tf.Session() as sess:
        # Restore variables from disk.

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./test' ,sess.graph)
        
        inputs = {"Datafile":  hparams.data_path}           # Get the data filepath 

        
        saver.restore(sess, hparams.save_path)    

        #tf.summary.scalar('loss', tf.reduce_sum(loss))

        dist = utils.get_dist_data(inputs)                  # Get array containing distance of the object node to router 
        dist = utils.normalize_dist(dist) 
        X = utils.get_rssi_data(inputs)                     # Get RSSI data. Shape =  num_data_points * num_nodes
        X = utils.normalize_data(X)
        dist_labels = dist

        length = len(dist_labels)                           # Get length of the dataset     
        train_length = int(length*0.8)                      # Train dataset length 
        test_length = int(length*0.2)                       # Test dataset length 
        
        X_train = X[0:train_length]                         # Train dataset (features)
        X_test = X[-test_length:]                           # Test dataset (features)

        dist_train = dist_labels[0:train_length]                   # Train dataset (labels)
        dist_test = dist_labels[-test_length:]                     # Test dataset (labels)    

        total_iter_test_error = 0
    
        test_epochs = (test_length-4)/hparams.batch_size
        accuracy_list = []
        
        for k in range(test_epochs):
            
            test_rssi = utils.getbatch_X(k,X_test)
            test_labels = utils.getbatch_labels(k,dist_test)
            test_labels = np.reshape(test_labels,(hparams.batch_size,1)) 
            
            accuracy_val, pred_val = sess.run(
                    [accuracy, prediction],
                    feed_dict={
                        input_rssi: test_rssi, 
                        distance: test_labels,
                        # batchsz: hparams.test_labels_length
                    }
                )
            #pred_val = pred_val*np.sqrt(1315.2341209347906) + 198.05518950451153
            #test_labels = test_labels#*np.sqrt(1315.2341209347906) + 198.05518950451153
            pred_val = pred_val*(np.sqrt(3928.4198316935376)) + 176.15052954295948
            test_labels = test_labels*(np.sqrt(1489.9552173615964)) + 176.15052954295948
            
            temp = np.stack((pred_val, test_labels), axis=-1)
            print temp

                                   
            accuracy_val = np.sum(accuracy_val)
            total_iter_test_error = accuracy_val + total_iter_test_error
            accuracy_list.append(total_iter_test_error)
            np.save('accuracy', accuracy_list)

        # answer = pd.DataFrame(answer)
        # answer.to_csv('predictions.csv')
        print "Testing Error ", total_iter_test_error/test_epochs


if hparams.test_bool:
    test()
if hparams.train_bool:
    train()




