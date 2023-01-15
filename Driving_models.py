from keras.models import Model
from keras.layers import Dense, Flatten, Input
import tensorflow as tf
from lib import data_split, batch_iter
import numpy as np


class TFRidge(object):
    """ Ridge Regression """

    def __init__(self, sess):
        self.sess = sess
        self.reuse = False
        self.variables = None
        self.weighted = False
        self.name = 'ridge'
        self.weight = None
        self.channels = [0, 1, 4, 5, 6, 7, 10, 11, 16, 17, 19, 20, 21, 25, 32, 33, 36, 37, 38, 39, 42, 43, 44, 45, 46,
                         47, 48, 49,
                         52, 53, 54, 55, 58, 59]

    def __call__(self, input):
        with tf.variable_scope(self.name) as scope:
            if self.reuse:
                scope.reuse_variables()
            x_fft = tf.signal.fft(tf.cast(input, tf.complex64))
            # sample 1025; freq 250
            x_psd = tf.real(x_fft * tf.conj(x_fft)) / 7500
            x_psd = x_psd[:, :, :3750]
            # theta band 4-7.5 Hz; Alpha band 7.5-12 Hz
            x_psd = tf.concat([tf.reduce_mean(x_psd[:, :, 60:115], axis=2, keep_dims=True),
                               tf.reduce_mean(x_psd[:, :, 116:180], axis=2, keep_dims=True)],
                              axis=2)
            x_psd = tf.reshape(x_psd, [-1, 30 * 2])
            x_feature = tf.reshape(x_psd[:, 0], [-1, 1])
            for i in range(1, len(self.channels)):
                x_feature = tf.concat([x_feature, tf.reshape(x_psd[:, self.channels[i]], [-1, 1])], axis=1)
            output = tf.layers.dense(x_feature, units=1, activation=None, use_bias=True)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output

    def set_weight(self, w, b):
        self.weight = [w, b]

    def eval(self, data):
        data_place = tf.placeholder(tf.float32, shape=[None, data[0].shape[1], data[0].shape[2]])
        label_place = tf.placeholder(tf.float32, shape=[None, 1])
        pre_y = self.__call__(data_place)
        # loss = tf.losses.mean_squared_error(label_place, pre_y)
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(label_place, pre_y)))) # RMSE

        if ~self.weighted:
            self.sess.run(self.variables[0].assign(self.weight[0]))
            self.sess.run(self.variables[1].assign(self.weight[1]))
        self.weighted = True

        test_pre, test_loss = self.sess.run([pre_y, loss],
                                            feed_dict={data_place: data[0], label_place: data[1]})

        print('test loss = {}'.format(test_loss))
        pre_avg = np.sum(test_pre) / len(test_pre)
        label_avg = np.sum(data[1]) / len(data[1])
        cc = np.corrcoef(test_pre.squeeze(), data[1].squeeze())
        print('pre_avg = {}, label_avg = {}, cc = {}'.format(pre_avg, label_avg, cc))
        return test_pre.squeeze(), test_loss, pre_avg, cc


class DensNet(object):

    def __init__(self, sess, input_shape, model_path, batch_size=10, learning_rate=8e-5):
        self.sess = sess
        self.batch_size = batch_size
        self.lr = learning_rate
        self.earlystop_threshold = 4
        self.save_path = model_path
        self.reuse = False
        self.name = 'dense'
        self.data = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1]])
        self.labels = tf.placeholder(tf.float32, [None, 1])
        self.variables = None
        self.channels = [0, 1, 4, 5, 6, 7, 10, 11, 16, 17, 19, 20, 21, 25, 32, 33, 36, 37, 38, 39, 42, 43, 44, 45, 46,
                         47, 48, 49,
                         52, 53, 54, 55, 58, 59]

    def __call__(self, input):
        with tf.variable_scope(self.name) as scope:
            if self.reuse:
                scope.reuse_variables()
            x_fft = tf.signal.fft(tf.cast(input, tf.complex64))
            # sample 1025; freq 250
            x_psd = tf.real(x_fft * tf.conj(x_fft)) / 7500
            x_psd = x_psd[:, :, :3750]
            # theta band 4-7.5 Hz; Alpha band 7.5-12 Hz
            x_psd = tf.concat([tf.reduce_mean(x_psd[:, :, 60:115], axis=2, keep_dims=True),
                               tf.reduce_mean(x_psd[:, :, 116:180], axis=2, keep_dims=True)],
                              axis=2)
            x_psd = tf.reshape(x_psd, [-1, 30 * 2])
            x_feature = tf.reshape(x_psd[:, 0], [-1, 1])
            for i in range(1, len(self.channels)):
                x_feature = tf.concat([x_feature, tf.reshape(x_psd[:, self.channels[i]], [-1, 1])], axis=1)
            # x_feature = tf.layers.batch_normalization(inputs=x_feature, axis=-1,
            #                                           training=True, trainable=True)
            x = tf.layers.dense(x_feature, units=50, activation=tf.nn.relu, use_bias=True)
            x = tf.layers.dense(x, units=50, activation=tf.nn.relu, use_bias=True)
            output = tf.layers.dense(x, units=1, activation=None, use_bias=True)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return output

    def train(self, data):
        data = data_split(data[0], data[1])
        train = data['train']
        validation = data['test']

        pre_y = self.__call__(self.data)
        # loss = tf.losses.mean_squared_error(self.labels, pre_y)
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels, pre_y)))) # RMSE

        start_vars = set(x.name for x in tf.global_variables())
        train_step = tf.train.AdamOptimizer(self.lr).minimize(loss, var_list=self.variables)
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        saver = tf.train.Saver(var_list=self.variables, max_to_keep=3)

        self.sess.run(tf.variables_initializer(var_list=self.variables + new_vars))

        pre_train_loss = 1.0e5
        min_val_loss = 1.0e5
        earlystop_cnt = 0

        for epoch in range(1000):
            batches = batch_iter(list(zip(train[0], train[1])), self.batch_size)
            iteration = 0
            batch_loss = 0
            for batch in batches:
                x, y = zip(*batch)
                _, train_loss, pre = self.sess.run([train_step, loss, pre_y],
                                                   feed_dict={self.data: x, self.labels: y})
                batch_loss += train_loss
                iteration += 1
            batch_loss /= iteration
            val_loss = self.sess.run(loss, feed_dict={self.data: validation[0], self.labels: validation[1]})

            if epoch % 10 == 0:
                print(
                    'epoch: {}, train loss = {}, val loss = {}'.format(epoch, batch_loss, val_loss))

            # early stoping
            if val_loss > min_val_loss:
                if batch_loss < pre_train_loss:
                    if earlystop_cnt >= self.earlystop_threshold:
                        print('early stopped on ' + str(epoch))
                        break
                    else:
                        earlystop_cnt += 1
                        print('overfitting warning: ' + str(earlystop_cnt))
                else:
                    earlystop_cnt = 0
            else:
                earlystop_cnt = 0
                min_val_loss = val_loss
                saver.save(self.sess, self.save_path + '/model.ckpt', global_step=epoch + 1)

            pre_train_loss = train_loss

    def eval(self, data):
        """ Evaluate DNN """
        pre_y = self.__call__(self.data)
        # loss = tf.losses.mean_squared_error(self.labels, pre_y)
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.labels, pre_y))))

        saver = tf.train.Saver(var_list=self.variables, max_to_keep=3)
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(self.save_path))

        test_pre, test_loss = self.sess.run([pre_y, loss],
                                            feed_dict={self.data: data[0], self.labels: data[1]})

        print('test loss = {}'.format(test_loss))
        pre_avg = np.sum(test_pre) / len(test_pre)
        label_avg = np.sum(data[1]) / len(data[1])
        cc = np.corrcoef(test_pre.squeeze(), data[1].squeeze())
        print('pre_avg = {}, label_avg = {}, cc = {}'.format(pre_avg, label_avg, cc))

        return test_pre.squeeze(), test_loss, pre_avg, cc
