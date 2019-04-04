from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import tensorflow as tf
#import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
sys.path.append('/home/carlo/workspace/networkqit')
import networkqit as nq
from scipy.linalg import expm, eigvalsh
from tqdm import tqdm

# Imperative part
G = nx.connected_caveman_graph(4,8)
#G = nx.disjoint_union(G,nx.connected_caveman_graph(2,3))
#N = len(G.nodes())
A = nx.to_numpy_array(G)

def benchmark():
   nr = np.array([32, 16, 8, 4])
   ers = np.outer(nr, nr-1)/2
   np.fill_diagonal(ers, nr*(nr-1))
   print(ers,nr)
   return nq.sbm(ers, nr)

#A = benchmark()
#A = nx.to_numpy_array(G)
A = np.loadtxt('/home/carlo/workspace/communityalg/data/Coactivation_matrix_bin.adj')[0:64, 0:64]
#A = nx.to_numpy_array(nx.random_partition_graph([32,16,8,4],1,0.05))
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


##### STOCHASTIC OPTIMIZATION #########
nepochs = 5000
refresh_frames = 10
batch_size = 16
beta_range = np.linspace(2, 0.005, 20)

def compute_avg_rel_entropy(self, beta, Lobs, rho, Amodel):
    # Compute the model Laplacian
    Lmodel = tf.matrix_diag(tf.reduce_sum(Amodel, axis=1)) - Amodel
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

def sample(pij, reparametrization, sigmoid_slope=200):
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
    return Amodel

with tf.device('/device:CPU:0'):
    Lobs = tf.placeholder(dtype=tf.float32, name='Lobs', shape=[N, N])
    rho = tf.placeholder(dtype=tf.float32, name='rho', shape=[N, N])
    beta = tf.placeholder(dtype=tf.float32, name='beta', shape=())
    eta = tf.placeholder(dtype=tf.float32, name='eta', shape=())

    # Declare optimization variables initialization
    pij0 = tf.random_uniform(shape=[N, N], minval=0.1, maxval=1, dtype=tf.float32)
    # Declare the optimization variables
    pij = tf.Variable(pij0, name='pij', dtype=tf.float32, trainable=True)
    pij = tf.reshape(tf.tile(pij, [batch_size,1]),[batch_size,N,N])
    # set 0 on diagonal
    pij = tf.matrix_set_diag(pij, tf.constant([0.0], shape=[batch_size, N]))
    # set 0 under the diagonal
    pij = tf.linalg.band_part(pij,0,-1) # np.triu(A,1)
    # copy the upper triangular to the lower triangular, reducing the number of 
    # free variables to N*(N-1)/2
    pij += tf.linalg.transpose(pij) # enforces symmetry

    Amodel = sample(pij,'sigmoid', sigmoid_slope=50)
    rel_entropy = average_relative_entropy(beta, Lobs, rho, Amodel)

    # Define the optimizer
    method = 'ADAM'
    if method is 'ADAM':
        global_step = tf.Variable(0.0, trainable=False)
        # every nepochs learning rate is decreased by 90%
        exp_decayed_learning_rate = tf.train.exponential_decay(learning_rate=eta,
                                                               global_step=global_step,
                                                               decay_steps=1000,
                                                               decay_rate=0.9,
                                                               staircase=False)

        optimizer = tf.train.AdamOptimizer(learning_rate = exp_decayed_learning_rate)
        train = optimizer.minimize(rel_entropy, global_step=global_step)
        
        # For gradient clipping
        #gvs = optimizer.compute_gradients(rel_entropy)
        #capped_gvs = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in gvs]
        #train = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    # Initialize the global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        collect_dict = {'rel_entropy': [], 
                        'gradients': [],
                        'sol': [],
                        'beta': []}

        # Prepare the execution of the computation graph
        sess.run(init)

        from drawnow import drawnow, figure
        # if global namespace, import plt.figure before drawnow.figure
        figure(figsize=(10, 6))
        # for b in tqdm(beta_range):
        iteration = 0
        brho = nq.compute_vonneuman_density(L=L, beta=1)
        for b in beta_range:
            feed_dict = {Lobs: L, beta: b, eta: 0.01, rho: brho}
            #for epoch in tqdm(range(nepochs), desc='beta=%g ' % (b) + 'eta,S=%s' % sess.run([exp_decayed_learning_rate,rel_entropy],feed_dict=feed_dict)):
            while True:
                tfvals = sess.run([train, exp_decayed_learning_rate, rel_entropy], feed_dict=feed_dict)
                iteration+=1
                print('\r','iteration=',iteration,'eta=',tfvals[1],end='')
                # Termination conditions
                collect_dict['rel_entropy'].append(tfvals[-1])
                if (iteration % refresh_frames) == 0: # check every 100 iteration
                    res_slope, res_intercept = np.polyfit(range(refresh_frames),collect_dict['rel_entropy'][-refresh_frames:],1)
                    if np.abs(res_slope) < 1E-4:
                        def draw_fig():
                            plt.subplot(2, 3, 1)
                            im = plt.imshow(A)
                            plt.colorbar(im,fraction=0.046, pad=0.04)
                            plt.title('Empirical network')
                            plt.grid(False)

                            plt.subplot(2, 3, 2)
                            A0 = sess.run(Amodel, feed_dict=feed_dict)[0]
                            ki = A.sum(axis=0)
                            im = plt.imshow(A0)
                            plt.title('Sampled network')
                            plt.colorbar(im,fraction=0.046, pad=0.04)
                            plt.grid(False)

                            plt.subplot(2, 3, 3)
                            im = plt.imshow(sess.run(pij, feed_dict=feed_dict)[0])
                            plt.colorbar(im,fraction=0.046, pad=0.04)
                            plt.title('Model pij(xi)')
                            plt.clim([0, 1])
                            plt.grid(False)

                            plt.subplot(2, 3, 4)
                            for i in range(batch_size):
                                A0 = sess.run(Amodel, feed_dict=feed_dict)[i]
                                L0 = np.diag(np.sum(A0, axis=0)) - A0
                                plt.semilogx(np.logspace(-4, 4, 50), nq.batch_compute_vonneumann_entropy(L0, np.logspace(-4, 4, 50)), 'r-', alpha=0.5)
                            plt.semilogx(np.logspace(-4, 4, 50), nq.batch_compute_vonneumann_entropy(L, np.logspace(-4, 4, 50)), 'b-')
                            plt.semilogx(b, nq.compute_vonneumann_entropy(L=L, beta=b), 'ko')
                            plt.semilogx(1, nq.compute_vonneumann_entropy(L=L, beta=1), 'k*')
                            plt.title('Spectral Entropies')
                            plt.xlabel('beta')
                            plt.ylabel('Entropy')
                            plt.suptitle(('beta=%g' % b))

                            plt.subplot(2,3,5)
                            plt.plot(collect_dict['rel_entropy'], '-')
                            plt.ylabel('<S(rho,sigma)>')
                            plt.xlabel('iteration')
                            plt.title('<S(rho,sigma)>')
                            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                        drawnow(draw_fig)
                        break
    plt.waitforbuttonpress()
    plt.show()
