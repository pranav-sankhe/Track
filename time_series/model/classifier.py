import tensorflow as tf  
import numpy as np 
import os 
import sys
sys.path.insert(0, '../data')
import utils
import pandas as pd
import random
import hparams

label_flag = 0
data = pd.read_csv(hparams.data_path)
acc_cm_list = []
acc_labels_list = []
acc_rms_list = []

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

def get_label_length(num_labels):

    inputs = {"Datafile":  hparams.data_path}           # Get the data filepath 
    
    dist = utils.get_dist_data(inputs)                  # Get array containing distance of the object node to router 
    labels = np.histogram(dist, bins=num_labels-1)[1]
    label_len = [] 
    for i in range(len(labels)-1):
        label_len.append(labels[i+1] - labels[i])
    
    return np.mean(label_len)

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
    
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_labels, logits=logits)
    # target_weights = tf.sequence_mask(
    #     hparams.batch_size, max_time, dtype=logits.dtype)
    # if hparams.TIME_MAJOR:
    #     target_weights = tf.transpose(target_weights)

    # loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(batch_size)
    return crossent 

def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
      gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
    gradient_norm_summary.append(
      tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

    return clipped_gradients, gradient_norm_summary, gradient_norm



def train():

    input_rssi = tf.placeholder(tf.float32, [hparams.batch_size, hparams.num_time, hparams.num_nodes])
    distance  = tf.placeholder(tf.int64, [hparams.batch_size])
    act_distance  = tf.placeholder(tf.int64, [hparams.batch_size])
    optimizer = tf.train.AdamOptimizer(hparams.LR)
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    # correct_answer = tf.placeholder(tf.int64, [hparams.batch_size])
    global label_flag

    logits = core_model(hparams.lstm_sizes, input_rssi, keep_prob, hparams.batch_size)
    loss = compute_loss(distance, logits)
    label_len = 0 
    if label_flag == 0:
        label_len = get_label_length(hparams.num_labels)
        label_flag = 1
         
    # test time accuracy calculation
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, distance)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # accuracy_cm = tf.add(tf.abs(tf.subtract(distance, prediction))*label_len, label_len/2.0) 
    # accuracy_cm = tf.reduce_mean(accuracy_cm)
   
    prediction = tf.add(prediction*label_len, label_len/2.0)            #convert prediction label to value reported
    accuracy_rms = tf.square(tf.subtract(prediction, act_distance))     
    accuracy_rms = tf.reduce_mean(accuracy_rms)
    
    # tf.summary.scalar('accuracy in cm', accuracy_cm)
    tf.summary.scalar('accuracy in rms', accuracy_rms)

    
    with tf.name_scope("compute_gradients"):
        # compute_gradients` returns a list of (gradient, variable) pairs
        params = tf.trainable_variables()

        for var in params:
            tf.summary.histogram(var.name, var)
        
        grads = tf.gradients(xs=params, ys=loss, colocate_gradients_with_ops=True)    # optimizer.compute_gradients(loss)
        clipped_grads, grad_norm_summary, grad_norm = gradient_clip(grads, max_gradient_norm=hparams.max_gradient_norm)
        grad_and_vars = zip(clipped_grads, params)


    
    global_step = tf.train.get_or_create_global_step()
    apply_gradient_op = optimizer.apply_gradients(grad_and_vars, global_step)
    
    tf.summary.scalar('loss', tf.reduce_sum(loss))


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
        

        saver = tf.train.Saver()
        
        inputs = {"Datafile":  hparams.data_path}           # Get the data filepath 
        
        dist = utils.get_dist_data(inputs)                  # Get array containing distance of the object node to router 
        actual_dist = np.copy(dist)
        X = utils.get_rssi_data(inputs)                     # Get RSSI data. Shape =  num_data_points * num_nodes
        X = utils.normalize_data(X)
        
        label_vals = create_labels(hparams.num_labels, dist)
        dist_labels = []
        for dist_val in dist: 
            dist_labels.append(get_label(dist_val, label_vals))
    
        

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

        act_dist_train = actual_dist[0:train_length]
        act_dist_test = actual_dist[-test_length:]

        
        epochs = (len(dist_train) - 4)/hparams.batch_size         # number of epochs for training 

        for j in range(hparams.num_iterations): 
            print ("Training:: iteration: ", j)
             
            
            for i in range(int(epochs)-1):
                print ("Training:: Epoch ", i)

                train_batch_rssi = utils.getbatch_X(i, X_train)     # get batched data (features)
                train_batch_labels = utils.getbatch_labels(i, dist_train)     # get batched data (labels)
                train_batch_labels = np.reshape(train_batch_labels,(hparams.batch_size))
                _, loss_val, summary = sess.run(
                    [apply_gradient_op, loss, merged],
                    feed_dict={
                        input_rssi: train_batch_rssi, 
                        distance: train_batch_labels,
                        act_distance: train_batch_labels,
                        keep_prob: hparams.keep_prob_
                    }
                ) 
                train_writer.add_summary(summary, j)

                print (np.sum(loss_val)/np.size(loss_val))

            total_iter_test_error = 0
            
            
            test_epochs = (test_length - 4)/(hparams.batch_size)
            # accuracy_list_cm = []  
            accuracy_list_rms = []
            accuracy_labels = []          
            for k in range(int(test_epochs)):

                test_rssi = utils.getbatch_X(k,X_test)

                test_labels = utils.getbatch_labels(k,dist_test)
                test_labels = test_labels.astype(np.int64)

                test_act_dist = utils.getbatch_labels(k,act_dist_test)
                test_act_dist = test_act_dist.astype(np.float32)

                accuracy_val, accuracy_rms_val= sess.run(
                        [accuracy,accuracy_rms],
                        feed_dict={
                            input_rssi: test_rssi, 
                            distance: test_labels,
                            act_distance: test_act_dist,
                            keep_prob: 1
                        }
                    )

                
                accuracy_labels.append(accuracy_val)
                # accuracy_list_cm.append(accuracy_in_cm)
                accuracy_list_rms.append(accuracy_rms_val)
            
            # acc_cm = np.mean(accuracy_in_cm)
            # print ("accuracy in cms is", acc_cm)
            # acc_cm_list.append(acc_cm)

            acc_rms = np.mean(accuracy_list_rms)
            acc_rms = np.sqrt(acc_rms)
            print ("accuracy rms is", acc_rms)
            acc_rms_list.append(acc_rms)

            
            acc_label = np.mean(accuracy_labels)
            print ("accuracy classification is", acc_label)
            acc_labels_list.append(acc_label)
            
            # np.save('accuracy_in_list', acc_cm_list)        
            np.save('accuracy_list', acc_labels_list)    
            np.save('accuracy_rms_list', acc_rms_list)    

            if (j== hparams.num_iterations - 1):
                save_path = saver.save(sess, "./saved_model.ckpt")    #Save trained model
                print("Model saved in path: %s" % save_path)

def test():
    input_rssi = tf.placeholder(tf.float32, [hparams.batch_size, hparams.num_time, hparams.num_nodes])
    distance  = tf.placeholder(tf.int64, [hparams.batch_size])
    optimizer = tf.train.AdamOptimizer(hparams.LR)
    keep_prob = tf.placeholder_with_default(1.0, shape=())
    # correct_answer = tf.placeholder(tf.int64, [hparams.batch_size])
    

    logits = core_model(hparams.lstm_sizes, input_rssi, keep_prob, hparams.batch_size)
    loss = compute_loss(distance, logits)

    # test time accuracy calculation
    prediction = tf.argmax(logits, 1)
    equality = tf.equal(prediction, distance)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        # Restore variables from disk.

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./test' ,sess.graph)
        
        inputs = {"Datafile":  hparams.data_path}   # Get the data filepath 

        
        saver.restore(sess, hparams.save_path)
        dist = utils.get_dist_data(inputs)                  # Get array containing distance of the object node to router 
        X = utils.get_rssi_data(inputs)                     # Get RSSI data. Shape =  num_data_points * num_nodes
        X = utils.normalize_data(X)
        
        label_vals = create_labels(hparams.num_labels, dist)
        dist_labels = []
        for dist_val in dist: 
            dist_labels.append(get_label(dist_val, label_vals))
    
        
        length = len(dist_labels)                                  #Get length of the dataset     
        train_length = int(length*0.8)                      # Train dataset length 
        test_length = int(length*0.2)                       # Test dataset length 
        
        X_train = X[0:train_length]                         # Train dataset (features)
        X_test = X[-test_length:]                           # Test dataset (features)
            
        dist_train = dist_labels[0:train_length]                   # Train dataset (labels)
        dist_test = dist_labels[-test_length:]                     # Test dataset (labels)    
        
        total_iter_test_error = 0

        
        test_epochs = (test_length - 4)/(hparams.batch_size)
                    
        for k in range(test_epochs):

            test_rssi = utils.getbatch_X(k,X_test)
            test_labels = utils.getbatch_labels(k,dist_test)
            test_labels = test_labels.astype(np.int64)
            
            accuracy_val = sess.run(
                    [accuracy],
                    feed_dict={
                        input_rssi: test_rssi, 
                        distance: test_labels,
                        keep_prob: 1
                    }
                )

            accuracy_list.append(accuracy_val)
            print (accuracy_val)
        np.save('accuracy_list', accuracy_list)       

if hparams.test_bool:
    test()
if hparams.train_bool:
    train()




