import tensorflow as tf
import numpy as np
import seaborn as sns; sns.set_style('whitegrid')
import os
from tqdm import tqdm
from plot_utils import *
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import stack, fully_connected as fc, flatten
from tensorflow.python.ops.distributions.bernoulli import Bernoulli
from tensorflow_probability.python.distributions import Gumbel
from tensorflow_probability.python.distributions import Categorical

# Define Directory Parameters
flags = tf.app.flags
flags.DEFINE_string('data_dir', os.getcwd() + '/data/', 'Directory for data')
flags.DEFINE_string('log_dir', os.getcwd() + '/log/', 'Directory for logs')
flags.DEFINE_string('results_dir', os.getcwd() + '/results/', 'Directory for results')

# Define Model Parameters
flags.DEFINE_integer('batch_size', 100, 'Minibatch size')
flags.DEFINE_integer('num_iters', 50000, 'Number of iterations')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('num_classes', 10, 'Number of classes')
flags.DEFINE_integer('num_cat_dists', 20, 'Number of categorical distributions') # num_cat_dists//num_calsses
flags.DEFINE_float('min_temp', 0.5, 'Minimum temperature')
flags.DEFINE_float('anneal_rate', 0.0001, 'Anneal rate')

FLAGS = flags.FLAGS


def gumbel_softmax(logits, temperature, hard=False):
    gumbel = Gumbel(loc=0., scale=1.)
    # NOTE: softmax over axis=2, which corresponds to the class dimension.
    y = tf.nn.softmax((logits + gumbel.sample(tf.shape(logits))) / temperature)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 2, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def encoder(x):
    """Make logits for variational proposal distribution.
    This is logits for K categorical distributions (K=num_cat_dists),
    where each categorical distribution is defined on N categories (N=num_classes).

    Parameters
    ----------
    x

    Returns
    -------
    logits: unnormalized log probability of shape (batch_size, num_cat_dists, num_classes)
    """
    net = stack(x, fc, [512, 256])
    return tf.reshape(fc(net, FLAGS.num_cat_dists * FLAGS.num_classes, activation_fn=None), [-1, FLAGS.num_cat_dists, FLAGS.num_classes])


def decoder(z):
    """Make reconstruction network.

    Parameters
    ----------
    z

    Returns
    -------
    Reconstruction distribution p(x|z;\theta) with Bernoulli distribution.
    Here, Bernoulli was chosen since pixel space is bounded by [0, 255].
    """
    net = stack(flatten(z), fc, [256, 512])
    logits = fc(net, 28*28, activation_fn=None)
    return Bernoulli(logits=logits)


def train():
    # Load MNIST data.
    data = input_data.read_data_sets(FLAGS.data_dir+'/MNIST', one_hot=True)

    # Create encoder graph.
    with tf.variable_scope("encoder"):
        inputs = tf.placeholder(tf.float32, shape=[None, 28*28], name='inputs')
        tau = tf.placeholder(tf.float32, shape=[], name='temperature')
        logits = encoder(inputs)
        z = gumbel_softmax(logits, tau, hard=False) # (batch_size, num_cat_dists, num_classes)

    # Create decoder graph.
    with tf.variable_scope("decoder"):
        p_x_given_z = decoder(z)

#    with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
#        categorical = Categorical(probs=np.ones(FLAGS.num_classes)/FLAGS.num_classes)
#        z = categorical.sample(sample_shape=[FLAGS.batch_size, FLAGS.num_cat_dists])
#        z = tf.one_hot(z, depth=FLAGS.num_classes)
#        p_x_given_z_eval = decoder(z)

    # Define loss function and train opeator.
    # NOTE: Categorically uniform prior p(z) is assumed.
    # NOTE: Also, in this case, KL becomes negative entropy.
    # NOTE: Summation becomes KLD over whole distribution q(z|x) since z is assumed to be elementwise independent.
    KL = - tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.nn.softmax(logits), logits=logits), axis=1)
    ELBO = tf.reduce_sum(p_x_given_z.log_prob(inputs), axis=1) - KL
    loss = tf.reduce_mean(-ELBO)
    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        temperature = 0.0
        for i in tqdm(range(1, FLAGS.num_iters)):
            np_x, np_y = data.train.next_batch(FLAGS.batch_size)
            _, np_loss = sess.run([train_op, loss], {inputs: np_x, tau: temperature})
            if i % 1000 == 0:
                temperature = np.maximum(FLAGS.min_temp, np.exp(-FLAGS.anneal_rate * i))
                print('Temperature updated to {}\n'.format(temperature))
            if i % 5000 == 1:
                print('Iteration {}\nELBO: {}\n'.format(i, -np_loss))

        # Plot results.
        x_mean = p_x_given_z.mean()
        batch = data.test.next_batch(FLAGS.batch_size)
        np_x = sess.run(x_mean, {inputs: batch[0], tau: FLAGS.min_temp})

        x_mean_eval = p_x_given_z_eval.mean()
        np_x_eval = sess.run(x_mean_eval)

        plot_squares(batch[0], np_x, np_x_eval, 8)


def main():
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.data_dir)
    tf.gfile.MakeDirs(FLAGS.results_dir)
    train()


if __name__=="__main__":
    main()
