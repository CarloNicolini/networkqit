from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
sys.path.append('/home/carlo/workspace/networkqit')
import networkqit as nq
from scipy.linalg import eigvalsh
from tqdm import tqdm
import logging
import bct

# Observation
G = nx.connected_caveman_graph(5,5)
A = nx.to_numpy_array(G)
N = len(A)
m = A.sum()
L = np.diag(A.sum(axis=0)) - A
lambda_obs = eigvalsh(L)

# make a generic plot of the observed spectral entropy
# plt.figure()
# plt.semilogx(np.logspace(-3, 3, 100),
#              nq.batch_compute_vonneumann_entropy(L, np.logspace(-3, 3, 100)))
# plt.grid(True)
# plt.show()

def average_relative_entropy( beta, Lobs, rho, Amodel):
    # Compute the model Laplacian
    Lmodel = tf.matrix_diag(tf.reduce_sum(Amodel, axis=1)) - Amodel
    # estimate is better!

    # Define loss function
    # Model energy
    Lmodelrho = tf.multiply(rho, Lmodel, name='Lmrho')
    #ELmodelrho = tf.multiply(rho, ELmodel, name='Lmrho')

    # Em is the ensemble average of <Trace[dot(rho,Lmodel)]>
    Em = tf.reduce_mean(tf.reduce_sum(Lmodelrho, name='TrLmrho', axis=[1, 2]))

    # Observation energy (observation eigenvalues don't need to be computed)
    Lobsrho = tf.multiply(Lobs, rho, name='Lmrho')
    # here we just make a sum, because we miss a dimension (it's N x N)
    Eo = tf.reduce_sum(Lobsrho, name='TrLmrho')

    # Model laplacian eigenvalues, shape is [batch_size,N]
    lm = tf.linalg.eigvalsh(Lmodel, name='lambda_model')
    # Observation laplacian eigenvalues, shape is [N,1]
    lo = tf.linalg.eigvalsh(Lobs, name='lambda_obs')
    # Model free energy, reduce logsumexp on last dimension, then average over
    # the ensemble
    Fm = - tf.reduce_mean(tf.reduce_logsumexp(-beta * lm, name='Fm', axis=1)) / beta
    # Observation free energy
    Fo = - tf.reduce_logsumexp(-beta * lo, name='Fo') / beta
    # Loglikelihood and relative entropy to train
    loglike = beta * (-Fm + Em)
    entropy = beta * (-Fo + Eo)
    rel_entropy = loglike - entropy
    return rel_entropy

def sample_ubcm(xi, reparametrization, sigmoid_slope=200):
    # Through einstein summation we perform outer product over all batches
    xij = tf.einsum('ij,ik->ijk', xi, xi) # a [batch_size,N,N] tensor
    
    # Edge probability pij as from the Undirected Binary Configuration Model
    pij = xij / (1.0 + xij)  # explicit broadcasting of 1.0
    pij = tf.abs(pij) # to enforce positivity

    # Generate a batch of NxN matrices, with symmetric random elements
    rij = tf.random_uniform([batch_size, N, N], minval=0.0, maxval=1.0, name='rij')
    rij = tf.linalg.band_part(rij,0,-1) # equivalent to keeping upper triangular
    rij += tf.linalg.transpose(rij)
    # set random matrices diagonal to zero, over all batches
    rij = tf.matrix_set_diag(rij, tf.constant([0.0], shape=[batch_size, N]))

    # Sample from the probabilities pij
    # slope models the continuity approximation of the discrete bernoulli
    if reparametrization is 'sigmoid':
        # this is like setting < to 0 and > to 1, with shape [batch_size, N, N]
        Amodel = tf.sigmoid(sigmoid_slope * (pij-rij))
    elif reparametrization is 'relaxedbernoulli':
        import tensorflow_probability as tfp
        Amodel = tfp.distributions.RelaxedBernoulli(temperature=1/sigmoid_slope, probs=pij).sample()
        Amodel = tf.linalg.band_part(Amodel,0, -1) # like np.triu(A) for batched matrices  
        Amodel += tf.linalg.transpose(Amodel)
    # Set the diagonal to zero
    Amodel = tf.matrix_set_diag(Amodel,  tf.constant([0.0], shape=[batch_size, N]))
    return pij,Amodel

##### STOCHASTIC OPTIMIZATION #########
nepochs = 50000
refresh_frames = 2000
batch_size = 512
beta_range = np.linspace(10, 0.02, 20)
# Deferred part
with tf.device('/device:CPU:0'):
    Lobs = tf.placeholder(dtype=tf.float32, name='Lobs', shape=[N, N])
    rho = tf.placeholder(dtype=tf.float32, name='rho', shape=[N, N])
    beta = tf.placeholder(dtype=tf.float32, name='beta', shape=())
    eta = tf.placeholder(dtype=tf.float32, name='eta', shape=())

    # Declare optimization variables initialization
    x0 = tf.random_uniform(shape=[batch_size, N], minval=0.0, maxval=1, dtype=tf.float32)
    # Declare the optimization variables
    xi = tf.Variable(x0, name='xi', dtype=tf.float32, trainable=True)
    # Get the probability matrix and the sampled adjacency matrices
    pij, Amodel = sample_ubcm(xi, 'sigmoid')
    # Compute the relative entropy
    rel_entropy = average_relative_entropy(beta, Lobs, rho, Amodel)

    method = 'ADAM'
    if method is 'ADAM':
        global_step = tf.Variable(0.0, trainable=False)
        # every nepochs learning rate is decreased by 90%
        exp_decayed_learning_rate = tf.train.exponential_decay(learning_rate=eta,
                                                               global_step=global_step,
                                                               decay_steps=5000,
                                                               decay_rate=0.9,
                                                               staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=eta)
        # For gradient clipping
        #gvs = optimizer.compute_gradients(rel_entropy)
        #capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        #train = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        train = optimizer.minimize(rel_entropy, global_step=global_step)

    # Initialize the global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        collect_dict = {'rel_entropy': [], 'gradients': [], 'sol': [], 'beta': []}

        # Prepare the execution of the computation graph
        sess.run(init)
        logging.info('==== Start model fitting ====')

        from drawnow import drawnow, figure
        # if global namespace, import plt.figure before drawnow.figure
        figure(figsize=(5, 10))
        # for b in tqdm(beta_range):
        iteration = 0
        for b in beta_range:
            xi = tf.Variable(x0, name='xi', dtype=tf.float32, trainable=True)
            brho = nq.compute_vonneuman_density(L=L, beta=b)
            feed_dict = {Lobs: L, beta: b, eta: 0.05, rho: brho}
            for epoch in tqdm(range(nepochs), desc='beta=%g eta=%g' % (b,sess.run(exp_decayed_learning_rate,feed_dict=feed_dict))):
                iteration+=1
                tfvals = sess.run([train, exp_decayed_learning_rate, rel_entropy], feed_dict=feed_dict)
                collect_dict['rel_entropy'].append(tfvals[-1])
                if (iteration % refresh_frames) == 0:
                    def draw_fig():
                        plt.subplot(4, 2, 1)
                        plt.imshow(A)
                        plt.title('Empirical network')
                        plt.colorbar()
                        plt.grid(False)

                        plt.subplot(4, 2, 2)
                        A0 = sess.run(Amodel, feed_dict=feed_dict)[0]
                        ki = A.sum(axis=0)
                        bounds, ixes = bct.grid_communities(bct.community_louvain(A0)[0])
                        plt.imshow(A0)
                        plt.title('Sampled network')
                        plt.colorbar()
                        plt.grid(False)

                        plt.subplot(4, 2, 3)
                        plt.imshow(np.outer(ki, ki) / ki.sum())
                        plt.clim([0, 1])
                        plt.colorbar()
                        plt.title('Empirical kikj/2m')
                        plt.grid(False)

                        plt.subplot(4, 2, 4)
                        plt.imshow(sess.run(pij, feed_dict=feed_dict)[0])
                        plt.colorbar()
                        plt.title('Model pij(xi)')
                        plt.clim([0, 1])
                        plt.grid(False)

                        plt.subplot(4, 2, 5)
                        A0 = np.squeeze(sess.run(Amodel, feed_dict=feed_dict)[0, :, :])
                        k0 = np.sum(A0, axis=0)
                        plt.plot(range(N), ki - k0, '.r')
                        plt.xlabel('node')
                        plt.ylabel('')
                        plt.title('Degree difference')

                        plt.subplot(4, 2, 6)
                        plt.plot(collect_dict['rel_entropy'][-nepochs:-1], '-')
                        plt.ylabel('<S(rho,sigma)>')
                        plt.xlabel('iteration')
                        plt.title('<S(rho,sigma)>')

                        plt.subplot(4, 2, 7)
                        plt.hist(sess.run(tf.get_default_graph().get_tensor_by_name("lambda_model:0"),
                                         feed_dict=feed_dict).flatten(), N, density=True, alpha=0.7)
                        plt.hist(lambda_obs, density=True, alpha=0.7)
                        plt.title('Laplacian eigs')
                        plt.xlabel('eigenvalues')
                        plt.ylabel('norm. frequency')
                        plt.legend(['Model', 'Obs'])

                        plt.subplot(4, 2, 8)

                        for i in range(batch_size):
                            A0 = sess.run(Amodel, feed_dict=feed_dict)[i]
                            L0 = np.diag(np.sum(A0, axis=0)) - A0
                            plt.semilogx(np.logspace(-4, 4, 50), nq.batch_compute_vonneumann_entropy(L0, np.logspace(-4, 4, 50)), 'r-', alpha=0.5)
                        plt.semilogx(np.logspace(-4, 4, 50), nq.batch_compute_vonneumann_entropy(L, np.logspace(-4, 4, 50)), 'b-')
                        plt.semilogx(b, nq.compute_vonneumann_entropy(L=L, beta=b), 'ko')
                        plt.title('Spectral Entropies')
                        plt.xlabel('beta')
                        plt.ylabel('Entropy')
                        plt.suptitle(('beta=%g' % b))
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    drawnow(draw_fig)

        logging.info('==== Done model fitting ====')
    plt.waitforbuttonpress()
    plt.show()
