import gensim
from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
#import keras
import csv
import itertools
import sklearn
from itertools import islice
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers import *
from sklearn.metrics.pairwise import cosine_similarity


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

batch_size = 32

cl_u = 300
# cl_u = number of convolutional units
d = 200
# d = word embedding size
k = 3
# k = word embedding window size for z_n
nq = 404290
# nq = number of questions
n = 100
# length of vector
model = Word2Vec.load('/home/cameron/CS/CSTensorFlow/mymodel')
#word_vectors = model.wv


# store the embeddings in a numpy array
"""
question_1 = False
question_2 = False
linecounter = 0
with open('/home/cameron/CS/CSTensorFlow/output/newOutput.txt') as f:
    for line in f:
        linecounter += 1
       # print(line)
        print(linecounter)
        counter = 0
        if question_1 is False:
            input_string1 = line
            question_1 = True
        else:
            input_string2 = line
            question_1 = False
            # there is a q1 and q2 now
            q1 = np.zeros(100, dtype = object)
            for word in input_string1.split():
                q1[counter] = model.wv[word]
                if counter > 98:
                    break
                counter += 1
            counter = 0
            q2 = np.zeros(100, dtype = object)
            for word in input_string2.split():
                if counter > 99:
                    break
                q2[counter] = model.wv[word]
                counter += 1


           #	 print(q1)
          #  print("linercounter ", + linecounter)

"""	

# shape is clu, k, d

# layer to compute the question-wide vectors
# produces features for each word in the question then sums them to create a question vector 
X1 = tf.placeholder(tf.float32, [None, 100, 200])
X2 = tf.placeholder(tf.float32, [None, 100, 200])
# currently x [Batch Size, n, d] y [k, d, cl_u]
Y = tf.placeholder(tf.float32, [3, 200, 300])
#x = tf.to_float(x, name='ToFloat')
W1 = tf.Variable(tf.zeros([3, 200, 300]), name='W1')
b = tf.Variable(tf.zeros([300]), name="b")


labels = []
with open('/home/cameron/CS/CSTensorFlow/train_labels.csv', 'rt') as infile:
            reader = csv.reader(infile, delimiter = ' ', quotechar = '|')
            for row in reader:
                labels.append(row)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print(sess.run(embedding_init, feed_dict={embedding_placeholder: embedding_matrix}))
    
    # 0 - 100; each iteration is next 100; first 1000 questions
    for i in range(1,11):
        # this mess creates the word embedding matrix for each question

        # upper limit on current batch index
        batch_counter = i * 100
        question_1 = False
        question_2 = False
        linecounter = 0
        batch = []
        with open('/home/cameron/CS/CSTensorFlow/output/newOutput.txt') as f:
            for line in islice(f, (i-1)*100, batch_counter):
                linecounter += 1
                print(linecounter)
                counter = 0
                
                if question_1 is False:
                    input_string1 = line
                    question_1 = True
                else:
                    input_string2 = line
                    question_1 = False
                    # there is a q1 and q2 now
                    q1 = np.zeros(100, dtype = object)
                    for word in input_string1.split():
                        q1[counter] = model.wv[word]
                        if counter > 98:
                            break
                        counter += 1
                    counter = 0

                    batch.append(q1)
                    q2 = np.zeros(100, dtype = object)
                    for word in input_string2.split():
                        if counter > 99:
                            break
                        q2[counter] = model.wv[word]
                        counter += 1
                    batch.append(q2)
        
        #print(batch[0])
        #print(batch[1])	
        #print(len(batch))
        #batch = np.asarray(batch)
        print(batch[0])
        #TODO not sure which batch I want to feed
        for tempi in range (1, 50):
            #train_step.run(feed_dict={X1: batch[2* tempi-1], X2: batch[2*tempi]})
            #feed_dict={X1: batch[2* tempi-1], X2: batch[2* tempi]}
            #X1 = batch[2*tempi -1]
            
            
            print("HELLO ABOVE THIS LINE IS X1")
            value1 = sess.run(tf.tanh(tf.nn.conv1d(X1, W1, stride = 1, padding = 'SAME') + b), feed_dict = {X1: batch[2* tempi-1]})
            value2.append(sess.run(tf.tanh(tf.nn.conv1d(X1, W1, stride = 1, padding = 'SAME') + b), feed_dict = {X2: batch[2* tempi]}))
            
            conv1 = tf.tanh(tf.nn.conv1d(X1, W1, stride = 1, padding = 'SAME') + b)
            conv2 = tf.tanh(tf.nn.conv1d(X2, W1, stride = 1, padding = 'SAME') + b)

            conv1flat = np.hstack(sess.run(value1))            
            conv2flat = np.hstack(sess.run(value2))
 
            cos_sim = cosine_similarity(conv1flat, conv2flat)

            print(cos_sim)
            print(conv_layer1)

            #then convolve q2
            #then take cosine similarity and adjust weights

            #redo the i and looping so it's 2i-1
        

        


#print(embedding_weights)

#free memory
del(model)
