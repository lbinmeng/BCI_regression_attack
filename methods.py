import numpy as np
import tensorflow as tf
import os


class CWL2(object):
    def __init__(self, sess, model, initial_c, batch_size, learning_rate, target, binary_search_step, input_shape,
                 max_iteration):
        self.sess = sess
        self.model = model
        self.initial_c = initial_c
        self.batch_size = batch_size
        self.lr = learning_rate
        self.target = target
        self.binary_search_step = binary_search_step
        self.shape = [batch_size, input_shape[0], input_shape[1]]
        self.max_iteration = max_iteration

        # variable to optimize
        modifier = tf.Variable(np.zeros(shape=self.shape), dtype=tf.float32)

        self.tdata = tf.Variable(np.zeros(self.shape), dtype=tf.float32, name='tdata')
        self.const = tf.Variable(np.zeros(self.batch_size), dtype=tf.float32, name='const')

        self.assign_tdata = tf.placeholder(tf.float32, shape=self.shape, name='assign_tdata')
        self.assign_const = tf.placeholder(tf.float32, shape=[batch_size], name='assign_const')

        # clip the example
        self.newdata = (tf.tanh(modifier + self.tdata) + 1) / 2
        self.oridata = (tf.tanh(self.tdata) + 1) / 2

        # prediction of model
        self.toutput = model(self.oridata)
        self.output = model(self.newdata)

        # L2 norm
        self.l2 = tf.reduce_sum(tf.square(self.newdata - self.oridata), axis=[1, 2])

        # Loss
        loss_pre = tf.maximum(self.toutput + self.target - self.output, np.asarray(0., dtype=np.float32))
        self.loss1 = tf.reduce_sum(self.const * loss_pre)
        self.loss2 = tf.reduce_sum(self.l2)
        self.loss = self.loss1 + self.loss2

        # Train variables
        start_vars = set(x.name for x in tf.global_variables())
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # Initial
        self.setup = []
        self.setup.append(self.tdata.assign(self.assign_tdata))
        self.setup.append(self.const.assign(self.assign_const))

        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, data):
        batch_size = self.batch_size
        l2 = []
        [adv, l2_temp] = self.attack_batch(data[:batch_size])
        l2.append(l2_temp)
        for i in range(1, len(data) // batch_size):
            print('Running CW attack on instance {} of {}'.format(i * batch_size, len(data)))
            # adv.append(self.attck_batch(data[i * batch_size:(i + 1) * batch_size]))
            [temp_adv, l2_temp] = self.attack_batch(data[i * batch_size:(i + 1) * batch_size])
            adv = np.concatenate([adv, temp_adv], axis=0)
            l2.append(l2_temp)

        if len(data) % batch_size != 0:
            last_idx = len(data) - (len(data) % batch_size)
            print('Running CW attack on instace {} of {}'.format(last_idx, len(data)))
            temp_data = np.zeros((batch_size,) + data.shape[1:])
            temp_data[:(len(data) % batch_size)] = data[last_idx:]
            [temp_adv, l2_temp] = self.attack_batch(temp_data)
            adv = np.concatenate([adv, temp_adv[:(len(data) % batch_size)]], axis=0)
            l2.append(l2_temp)
        return np.array(adv), np.mean(np.array(l2))

    def attack_batch(self, data):
        batch_size = self.batch_size

        data = np.clip(data, 0, 1)
        # convert to tanh-space
        data = (data * 2) - 1
        data = np.arctanh(data)

        lower_bound = np.zeros(batch_size)
        const = np.ones(batch_size) * self.initial_c
        upper_bound = np.ones(batch_size) * 1e10

        # best l2, score, instance
        o_bestl2 = [1e10] * batch_size
        o_bestscore = [-1] * batch_size
        o_bestattack = np.copy(data)

        l2 = 0
        for step in range(self.binary_search_step):
            self.sess.run(self.init)
            batch = data[:batch_size]

            bestl2 = [1e10] * batch_size
            bestscore = [-1] * batch_size
            print('Binary search step {} of {}'.format(step, self.binary_search_step))

            self.sess.run(self.setup, {self.assign_tdata: batch, self.assign_const: const})

            for iteration in range(self.max_iteration):
                _, l, l2, pred_adv, pred, ndata, c = self.sess.run(
                    [self.train_step, self.loss, self.l2, self.output, self.toutput, self.newdata, self.const])

                if iteration % 50 == 0:
                    print(
                        'Iteration {} of {}: loss={:.3g} l2={:.3g} pred_adv={:.3g} pred={:.3g} c={:.3g}'.format(
                            iteration,
                            self.max_iteration, l,
                            np.mean(l2),
                            np.mean(pred_adv),
                            np.mean(pred),
                            np.mean(c)))

                # adjust the best result found so far
                for e, (dst, pre_a, pre, nd) in enumerate(zip(l2, pred_adv, pred, ndata)):
                    if dst < bestl2[e] and pred_adv[e] >= pred[e] + self.target:
                        bestl2[e] = dst
                        bestscore[e] = pre_a
                    if dst < o_bestl2[e] and pred_adv[e] >= pred[e] + self.target:
                        o_bestl2[e] = dst
                        o_bestscore[e] = pre_a
                        o_bestattack[e] = nd

            # adjust the constant
            for e in range(batch_size):
                if pred_adv[e] >= pred[e] + self.target and bestscore[e] != -1:
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                else:
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

            print('Sucessfully generated adversarial examples on {} of {} instance.'.format(sum(upper_bound < 1e9),
                                                                                            batch_size))
            o_bestl2 = np.array(o_bestl2)
            mean = np.mean(o_bestl2[o_bestl2 < 1e9])
            l2 = mean
            print('Mean successful distortion: {:.4g}'.format(mean))

        # return the best solution found
        o_bestl2 = np.array(o_bestl2)
        return o_bestattack, l2


class LeastLikeClass(object):
    """ Iterative Least-Likely Class Method for Regression """

    def __init__(self, sess, model, target, eps, alpha, iteration, clip_min=0.0, clip_max=1.0):
        self.sess = sess
        self.model = model
        self.target = target
        self.eps = eps
        self.alpha = alpha
        self.iteration = iteration
        self.clip_min = clip_min
        self.clip_max = clip_max

    def attack(self, data):
        x = tf.placeholder(tf.float32, shape=[None, data[0].shape[1], data[0].shape[2]], name='x')
        y = tf.placeholder(tf.float32, shape=[None, data[1].shape[1]], name='y')
        ori = tf.placeholder(tf.float32, shape=[None, data[0].shape[1], data[0].shape[2]], name='x_ori')

        data[0] = np.clip(data[0], self.clip_min, self.clip_max)

        y_pred = self.model(x)
        loss = tf.maximum(y + self.target - y_pred, np.asarray(0., dtype=np.float32))

        grad, = tf.gradients(loss, x)
        perturbation = self.alpha * tf.sign(grad)
        perturbation_clip = tf.clip_by_value(ori - (x - perturbation), -self.eps, self.eps)
        adv_x = ori - perturbation_clip

        adv = self.sess.run(adv_x, feed_dict={x: data[0], y: data[1], ori: data[0]})
        adv = np.clip(adv, self.clip_min, self.clip_max)

        for iter in range(1, self.iteration):
            adv = self.sess.run(adv_x, feed_dict={x: adv, y: data[1], ori: data[0]})
            adv = np.clip(adv, self.clip_min, self.clip_max)

        l2 = np.mean(np.sum(np.square(adv - data[0]), axis=(1, 2)))
        print('Mean successful distortion: {:.4g}'.format(l2))

        return adv, l2
