# Code obtained from https://muratsensoy.github.io/gen.html

import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import torch
from skimage import io
from tensorflow_probability import distributions as tfd
from os.path import dirname, realpath

# Add root directory to path
file_path = realpath(__file__)
file_dir = dirname(file_path)
parent_dir = dirname(file_dir)
sys.path.append(parent_dir)
from dataset import load_data
from pathlib import Path

K = 52  # number of classes
HEIGHT = 274
WIDTH = 174
CHANNELS = 3


# define some utility functions
def var(name, shape, init=None):
    if init is None:
        init = tf.truncated_normal_initializer(stddev=(2 / shape[0]) ** 0.5)
    return tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                           initializer=init)


def conv(Xin, f, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(Xin, f, strides, padding)


def max_pool(Xin, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    return tf.nn.max_pool(Xin, ksize, strides, padding)


def exp_evidence(logits):
    return tf.exp(logits)


def KL(alpha):
    K = alpha.get_shape()[-1].value
    beta = tf.constant(np.ones((1, K)), dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha, axis=1, keep_dims=True)
    S_beta = tf.reduce_sum(beta, axis=1, keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha), axis=1, keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta), axis=1, keep_dims=True) - tf.lgamma(S_beta)

    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)

    kl = tf.reduce_sum((alpha - beta) * (dg1 - dg0), axis=1, keep_dims=True) + lnB + lnB_uni
    return kl


def calc_entropy(p):
    return (-p * np.log(p + 1e-8)).sum(1)


def disc(x, name='disc', K=52):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x_inp = tf.reshape(x, [-1, HEIGHT, WIDTH, CHANNELS])

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

        x = tf.layers.conv2d(inputs=x_inp, kernel_size=5, filters=20,
                             activation=tf.nn.relu, padding='VALID',
                             kernel_regularizer=regularizer,
                             kernel_initializer=tf.variance_scaling_initializer())
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

        x = tf.layers.conv2d(inputs=x, kernel_size=5, filters=50,
                             activation=tf.nn.relu, padding='VALID',
                             kernel_regularizer=regularizer,
                             kernel_initializer=tf.variance_scaling_initializer())
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

        x = tf.layers.flatten(x)

        x = tf.layers.dense(inputs=x, units=500, activation=tf.nn.relu,
                            kernel_regularizer=regularizer,
                            kernel_initializer=tf.variance_scaling_initializer())

        x = tf.layers.dense(inputs=x, units=K,
                            kernel_regularizer=regularizer,
                            kernel_initializer=tf.variance_scaling_initializer())
        return x


layers_g = [{'filters': 143028, 'kernel_size': [1, 1], 'strides': [1, 1], 'padding': 'valid'},
            # {'filters': 434613, 'kernel_size': [1, 1], 'strides': [1, 1], 'padding': 'valid'},
            # {'filters': 434613, 'kernel_size': [1, 1], 'strides': [1, 1], 'padding': 'valid'}
            ]


def imgen(x):
    if len(x.get_shape()) == 2:
        m = x.get_shape()[1]
        layer = tf.reshape(x, [-1, 1, 1, m])
    else:
        layer = x

    depth = len(layers_g)
    for i in range(depth):
        layer_config = layers_g[i]
        is_output = ((i + 1) == depth)

        conv2d = tf.layers.conv2d_transpose(
            layer,
            filters=layer_config['filters'],
            kernel_size=layer_config['kernel_size'],
            strides=layer_config['strides'],
            padding=layer_config['padding'],
            activation=tf.nn.tanh if is_output else None,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            name='layer_' + str(i))

        if is_output:
            layer = conv2d
        else:
            norm = tf.layers.batch_normalization(conv2d, training=True)
            lrelu = tf.nn.leaky_relu(norm)
            layer = lrelu

    # [M, img_size, img_size, img_channels]
    output = tf.identity(layer, name='generated_images')
    return output


def encoder(x, n=100):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x = tf.reshape(x, [-1, HEIGHT, WIDTH, CHANNELS])
        x = tf.layers.conv2d(x, 20, 5, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)

        x = tf.layers.conv2d(x, 50, 5, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)

        x = tf.layers.flatten(x)

        loc = tf.layers.dense(x, n)
        scale = tf.layers.dense(x, n, tf.nn.softplus)
        code = tfd.MultivariateNormalDiag(loc, scale).sample()
    return code


def make_prior(code_size):
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)


def decoder(code):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        recon = imgen(code)
    return recon


def gen(code):
    with tf.variable_scope('gen', reuse=tf.AUTO_REUSE):
        n = tf.shape(code)[0]
        m = code.get_shape()[1]
        x = tf.concat((tf.random_normal(shape=(n, 2)), code), 1)
        x = tf.layers.dense(x, 32, tf.nn.leaky_relu)
        x = tf.layers.dense(x, 32, tf.nn.leaky_relu)
        x = tf.layers.dense(x, 32, tf.nn.leaky_relu)
        std = tf.layers.dense(x, m, tf.nn.softplus)
    return std


def diz(x):  # the discriminator in latent space
    with tf.variable_scope('diz', reuse=tf.AUTO_REUSE):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        x = tf.layers.dense(x, 32, tf.nn.leaky_relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
        x = tf.layers.dense(x, 32, tf.nn.leaky_relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
        x = tf.layers.dense(x, 32, tf.nn.leaky_relu, kernel_regularizer=regularizer, bias_regularizer=regularizer)
        x = tf.layers.dense(x, 1, kernel_regularizer=regularizer, bias_regularizer=regularizer)
    return x


def wloss(logits, maximize=True):
    labels = tf.ones_like(logits) if maximize else tf.zeros_like(logits)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))


def autoencoder(X=None, n=100):
    if X is None:
        X = tf.placeholder(shape=[None, HEIGHT * WIDTH * CHANNELS], dtype=tf.float32)
    code = encoder(X, n)

    std = gen(code)
    pdf = tfd.MultivariateNormalDiag(loc=code, scale_diag=(std + 1e-3))
    fake = pdf.sample()

    rlogits = diz(code)
    r_p = tf.nn.sigmoid(rlogits)

    flogits = diz(fake)
    f_p = tf.nn.sigmoid(flogits)

    recon = decoder(code)
    Xfake = decoder(fake)

    real_logits = disc(X, 'disc0')
    real_p = tf.nn.sigmoid(real_logits)

    fake_logits = disc(Xfake, 'disc0')
    fake_p = tf.nn.sigmoid(fake_logits)

    prior = make_prior(code_size=n)

    kl = -tf.reduce_mean(prior.log_prob(code))
    kl_fake = -tf.reduce_mean(prior.log_prob(fake))

    ae_vars = [v for v in tf.trainable_variables() if 'encoder/' in v.name or 'decoder/' in v.name]
    gen_vars = [v for v in tf.trainable_variables() if 'gen/' in v.name]
    disc_vars = [v for v in tf.trainable_variables() if 'disc0/' in v.name]
    diz_vars = [v for v in tf.trainable_variables() if 'diz/' in v.name]

    loss_diz = tf.reduce_mean(-tf.log(r_p + 1e-8)) + tf.reduce_mean(-tf.log(1 - f_p + 1e-8))

    loss_disc = tf.reduce_mean(-tf.log(real_p + 1e-8)) + tf.reduce_mean(-tf.log(1 - fake_p + 1e-8))

    loss_gen = tf.reduce_mean(-tf.log(1 - fake_p + 1e-8)) + tf.reduce_mean(-tf.log(f_p + 1e-8))

    recon = tf.layers.flatten(recon)

    rec_loss = tf.reduce_mean(tf.reduce_sum(tf.square(recon - X), 1) + 1e-4) + 0.1 * kl
    rec_loss += wloss(flogits, False)
    rec_step = tf.train.AdamOptimizer().minimize(rec_loss, var_list=ae_vars)

    diz_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss_diz, var_list=diz_vars)
    disc_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss_disc, var_list=disc_vars)
    gen_step = tf.train.RMSPropOptimizer(1e-4).minimize(loss_gen, var_list=gen_vars)

    rec_step = tf.group([rec_step for _ in range(10)])

    step = tf.group([rec_step, diz_step, disc_step, gen_step])
    return X, Xfake, code, recon, rec_loss, step


def misleading_alpha(alpha, y):
    K = y.get_shape()[-1].value
    indices = tf.where(tf.equal(y, tf.constant(0, dtype=tf.float32)))
    alp = tf.gather_nd(alpha, indices)
    alp = tf.reshape(alp, [-1, K - 1])
    return alp


def loss_fn(Y, evidence, real_p, fake_p):
    disc_loss = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(real_p + 1e-5), axis=1) +
                               tf.reduce_sum(-Y * tf.log((1.0 - fake_p) + 1e-5), axis=1))
    alp = misleading_alpha(evidence + 1, Y)
    disc_loss += tf.reduce_mean(KL(alp))
    return disc_loss


# train LeNet network
def LeNet_EDL_GEN(logits2evidence=exp_evidence, lmb=0.005, K=52):
    g = tf.Graph()
    with g.as_default():
        X = tf.placeholder(shape=[None, HEIGHT * WIDTH * CHANNELS], dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=[None, K], dtype=tf.float32, name='Y')
        adv_eps = tf.placeholder(dtype=tf.float32)

        recon = None
        _, X_fake, code, recon, rec_loss, step_gen = autoencoder(X, n=100)

        real_logits = disc(X, K=K)
        real_p = tf.nn.sigmoid(real_logits)

        fake_logits = disc(X_fake, K=K)
        fake_p = tf.nn.sigmoid(fake_logits)

        evidence = logits2evidence(real_logits)
        tf.identity(evidence, name='evidence_out')

        alpha = evidence + 1

        u = K / tf.reduce_sum(alpha, axis=1, keep_dims=True)  # uncertainty
        tf.identity(u, name='uncertainty_out')

        prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True)
        tf.identity(prob, name='prob_out')

        var_disc = [v for v in tf.trainable_variables() if 'disc/' in v.name]

        l2_loss = tf.losses.get_regularization_loss()

        disc_loss = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(real_p + 1e-8), axis=1) +
                                   tf.reduce_sum(-(1 - Y) * tf.log(1.0 - real_p + 1e-8), axis=1) +
                                   tf.reduce_sum(-tf.log((1.0 - fake_p) + 1e-8), axis=1))

        disc_loss = loss_fn(Y, evidence, real_p, fake_p)

        step_disc = tf.train.AdamOptimizer().minimize(disc_loss + l2_loss, var_list=var_disc)

        loss_grads = tf.gradients(disc_loss, X)[0]
        adv_x = X + adv_eps * tf.sign(loss_grads)

        step = tf.group([step_disc, step_gen])

        # Calculate accuracy
        pred = tf.argmax(real_logits, 1)
        truth = tf.argmax(Y, 1)
        match = tf.reshape(tf.cast(tf.equal(pred, truth), tf.float32), (-1, 1))
        acc = tf.reduce_mean(match)

        total_evidence = tf.reduce_sum(evidence, 1, keepdims=True)
        mean_ev = tf.reduce_mean(total_evidence)
        mean_ev_succ = tf.reduce_sum(tf.reduce_sum(evidence, 1, keepdims=True) * match) / tf.reduce_sum(match + 1e-20)
        mean_ev_fail = tf.reduce_sum(tf.reduce_sum(evidence, 1, keepdims=True) * (1 - match)) / (
                    tf.reduce_sum(tf.abs(1 - match)) + 1e-20)

        return g, step, X, Y, adv_eps, adv_x, recon, prob, acc, disc_loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail, real_logits, X_fake


def get_dataset(images_dir, labels_path):
    def process_row(r):
        image_path = images_dir + '/' + str(r[0]) + '.jpg'
        label = r[1]
        label_one_h = [0] * K
        label_one_h[label] = 1

        return io.imread(image_path).flatten(), label_one_h

    # Read labels and get images
    labels = pd.read_csv(labels_path)
    _x = []
    _y = []
    for r in labels.values:
        rsult = process_row(r)
        _x.append(rsult[0])
        _y.append(rsult[1])
    return np.asarray(_x), np.asarray(_y)


def next_batch(data, labels, i, bsz):
    i = i % (data.shape[0] // bsz)
    return data[i * bsz:(i + 1) * bsz], labels[i * bsz:(i + 1) * bsz], np.arange(data.shape[0])[i * bsz:(i + 1) * bsz]


def train():
    g, step, X, Y, adv_eps, adv_x, recon, prob, acc, loss, u, evidence, mean_ev, mean_ev_succ, mean_ev_fail, logits, X_fake = LeNet_EDL_GEN(
        exp_evidence)

    sess = tf.Session(graph=g)
    with g.as_default():
        sess.run(tf.global_variables_initializer())

        print('calling load data....')
        train_loader, test_loader = load_data(root_dir='../')
        # x_test = []
        # y_test = []
        #
        # for batch_idx, (data, target) in enumerate(test_loader):
        #     flattened_data = torch.flatten(data, start_dim=1)
        #     x_test += flattened_data
        #     new_labels = []
        #     for batch_ex in target:
        #         _nl = [0] * K
        #         _nl[batch_ex.item()] = 1
        #         new_labels.append(_nl)
        #     y_test += new_labels

        print('Data loading complete. Starting training...')
        for epoch in range(0, 20):

            # Run Training
            num_batches = len(train_loader.dataset) / train_loader.batch_size
            for batch_idx, (data, target) in enumerate(train_loader):
                new_labels = []
                for batch_ex in target:
                    _nl = [0]*K
                    _nl[batch_ex.item()] = 1
                    new_labels.append(_nl)

                flattened_data = torch.flatten(data, start_dim=1)
                feed_dict = {X: flattened_data, Y: new_labels}

                sess.run(step, feed_dict)
                print('epoch %d - %d%%) ' % (epoch + 1, (100 * (batch_idx + 1)) // num_batches),
                      end='\r' if batch_idx < num_batches-1 else '')
            print('Finished training, now running testing...')

            # Run testing
            batch_accs = []
            batch_succs = []
            batch_fails = []
            for batch_idx, (data, target) in enumerate(test_loader):
                new_labels = []
                for batch_ex in target:
                    _nl = [0]*K
                    _nl[batch_ex.item()] = 1
                    new_labels.append(_nl)
                flattened_data = torch.flatten(data, start_dim=1)

                test_acc, test_succ, test_fail = sess.run([acc, mean_ev_succ, mean_ev_fail],
                                                          feed_dict={X: flattened_data, Y: new_labels})

                batch_accs.append(test_acc)
                batch_succs.append(test_succ)
                batch_fails.append(test_fail)

            print('Test accuracy: ', np.mean(batch_accs))

        inputs_dict = {"X": X, "Y": Y}
        outputs_dict = {"prob": prob, "u": u}
        save_dir = 'saved_model'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        tf.saved_model.simple_save(sess, save_dir, inputs_dict, outputs_dict)


if __name__ == '__main__':
    train()
