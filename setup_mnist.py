## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            files = ["train-images-idx3-ubyte.gz",
                     "t10k-images-idx3-ubyte.gz",
                     "train-labels-idx1-ubyte.gz",
                     "t10k-labels-idx1-ubyte.gz"]
            for name in files:

                urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
        VALIDATION_SIZE = 5000
        
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]



class MNISTPrediction:
  def __init__(self, sess, use_log = False):
    self.sess = sess
    self.use_log = use_log
    self.img = tf.placeholder(tf.float32, (None, 28*28))
    self.softmax_tensor = tf.import_graph_def(
            sess.graph.as_graph_def(),
            input_map={'Placeholder:0': tf.reshape(self.img,((-1, 784))), 'dropout/Placeholder:0':1.0},
            return_elements=['fc3/add:0'])
  def predict(self, dat):
    dat = np.squeeze(dat)
    # scaled = (0.5 + dat) * 255
    # scaled = dat.reshape((1,) + dat.shape)
    scaled = dat.reshape(-1, 28*28)
    # print(scaled.shape)
    predictions = self.sess.run(self.softmax_tensor,
                         {self.img: scaled})
    predictions = np.squeeze(predictions)
    return predictions
    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()
    top_k = predictions.argsort()#[-FLAGS.num_top_predictions:][::-1]
    return top_k[-1]


CREATED_GRAPH = False
class MNISTModel:
  def __init__(self, sess, use_log = False):
    self.num_channels = 1
    self.image_size = 28
    self.num_labels = 10
    global CREATED_GRAPH
    self.sess = sess
    self.use_log = use_log
    if not CREATED_GRAPH:
      create_graph()
      CREATED_GRAPH = True
    self.model = MNISTPrediction(sess, use_log)

  def predict(self, img):
    # scaled = (0.5+tf.reshape(img,((299,299,3))))*255
    # scaled = (0.5+img)*255
      # check if a shape has been specified explicitly
    logit_tensor = tf.import_graph_def(
      self.sess.graph.as_graph_def(),
      input_map={'Placeholder:0': tf.reshape(img, (-1, 784)), 'dropout/Placeholder:0':1.0},
      return_elements=['fc3/add:0'])
    softmax_tensor = tf.nn.softmax(logit_tensor)
    return softmax_tensor[0]

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  #with tf.gfile.FastGFile('/tmp/zoo/zoo_frozen.pb', 'rb') as f:
  with tf.gfile.FastGFile('/tmp/zoo/zoo_frozen_8bit.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    #for line in repr(graph_def).split("\n"):
    #  if "tensor_content" not in line:
    #    print(line)
    _ = tf.import_graph_def(graph_def, name='')

