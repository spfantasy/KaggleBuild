import tensorflow as tf
import math
import numpy as np


class DAELayer:
    def __init__(self, layersize, input_tensor, is_training, max_epoch=20, batch_size=4, name=None):
        '''
        defines a layer to layer autoencoder
        @param[layersize] provide (dim_input, dim_output) tuple
        @param[input_tensor] if the input tensor is given, use it as a stream
                            else, generate a placeholder
        '''
        # extract params
        self.name = name
        input_size, output_size = layersize
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.is_training = is_training
        # get input tensor
        self.input = input_tensor
        if self.input is None:
            self.input = tf.placeholder(dtype=tf.float32,
                                        shape=[input_size])
        self.construct_encoder(input_size, output_size)
        if is_training:
            self.construct_decoder(input_size, output_size)
            self.construct_evaluator()

    def construct_encoder(self, input_size, output_size):
        with tf.variable_scope(self.name + "encoder", reuse=tf.AUTO_REUSE):
            self.w = tf.get_variable(
                name='weight',
                shape=[input_size, output_size],
                dtype=tf.float32,
                initializer=tf.initializers.orthogonal,
                trainable=True,
            )
            self.b = tf.get_variable(
                name='bias',
                shape=[1, output_size],
                dtype=tf.float32,
                initializer=tf.initializers.zeros,
                trainable=True,
            )
            if self.is_training:
                dropout = tf.nn.dropout(self.input, keep_prob=.8)
            else:
                dropout = tf.nn.dropout(self.input, keep_prob=1)
            self.logits = tf.matmul(dropout, self.w) + self.b
#             self.output = tf.sigmoid(self.logits)
            self.output = tf.tanh(self.logits)
#             self.output = self.logits

    def construct_decoder(self, input_size, output_size):
        with tf.variable_scope(self.name + "decoder", reuse=tf.AUTO_REUSE):
            self.decode_w = tf.get_variable(
                name='weight',
                shape=[output_size, input_size],
                dtype=tf.float32,
                initializer=tf.initializers.orthogonal,
                trainable=True,
            )
            self.decode_b = tf.get_variable(
                name='bias',
                shape=[1, input_size],
                dtype=tf.float32,
                initializer=tf.initializers.zeros,
                trainable=True,
            )
            self.decode_logits = tf.matmul(
                self.output, self.decode_w) + self.decode_b

    def construct_evaluator(self):
        with tf.variable_scope(self.name + "eval", reuse=tf.AUTO_REUSE):
            self.cost = tf.reduce_mean(
                tf.squared_difference(self.input, self.decode_logits))
            self.train_step = tf.train.AdamOptimizer().minimize(self.cost)

    def train(self, feed_tensor, sess, data, verbose):
        for epoch in range(self.max_epoch):
            # epoch training
            np.random.shuffle(data)
            # batch training
            for ptr in range(0, len(data), self.batch_size):
                sess.run(self.train_step, feed_dict={
                         feed_tensor: data[ptr:ptr + self.batch_size]})
            # batch accuracy testing
            MSE = self.cost.eval(feed_dict={feed_tensor: data})
            if verbose:
                print("Training@[{}][EPOCH{:0>3d}][MSE = {}]".format(
                    self.name, epoch, MSE))

    def __repr__(self):
        return "AutoEncoder Layer[{}][{}->{}]".format(self.name, self.input.shape[-1], self.output.shape[-1])


class StackedDAE:
    def __init__(self, input_tensor, shape, is_training, path='./tmp/model.ckpt'):
        self.DAELayers = []
        self.path = path
        prev_input = input_tensor
        for i, (front, end) in enumerate(zip(shape[:-1], shape[1:])):
            self.DAELayers.append(
                DAELayer(
                    layersize=(front, end),
                    input_tensor=prev_input,
                    is_training=is_training,
                    name='DAE_Layer_{}'.format(i)
                )
            )
            prev_input = self.DAELayers[-1].output
        self.input = input_tensor
        self.output = self.DAELayers[-1].output

    def train(self, data, verbose=True):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for layer in self.DAELayers:
                layer.train(feed_tensor=self.input, sess=sess,
                            data=data, verbose=verbose)
            save_path = saver.save(sess, self.path)
            if verbose:
                print("Model saved in path: {}".format(save_path))

    def encode(self, data):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.path)
            return self.output.eval(feed_dict={self.input: data})

    def __getitem__(self, index):
        try:
            obj = self.DAELayers[index]
            return obj
        except:
            raise IndexError("This StackedDAE has only {} layer(s) while you are indexing No.{} layer".format(
                len(self.DAELayers), index))
