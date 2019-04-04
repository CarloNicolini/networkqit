from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
sys.path.append('/home/carlo/workspace/networkqit')
import networkqit as nq
from scipy.linalg import expm, eigvalsh
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

# plt.style.use('fivethirtyeight')

tf.reset_default_graph()

# Imperative part
#G = nx.karate_club_graph()
G = nq.ring_of_cliques(5,5)
#N = len(G.nodes())
A = nx.to_numpy_array(G)

# def benchmark():
#    ers = np.array([[250, 20], [40, 100]])
#    #ers = np.kron(ers,np.array([[1,0.8],[0.8,0.5]]))
#    nr = [32, 48]
#    return nq.sbm(ers,nr)
#A = benchmark()
#A = nx.to_numpy_array(G)
#A = np.loadtxt(
#    '/home/carlo/workspace/communityalg/data/Coactivation_matrix_bin.adj')[0:64, 0:64]
N = len(A)
m = A.sum()
L = np.diag(A.sum(axis=0)) - A

lambda_obs = eigvalsh(L)
# make a generic plot of the observed spectral entropy
plt.figure()
plt.semilogx(np.logspace(-3, 3, 100),
             nq.batch_compute_vonneumann_entropy(L, np.logspace(-3, 3, 100)))
plt.grid(True)
plt.show()


##### STOCHASTIC OPTIMIZATION #########
nepochs = 50000
refresh_frames = 5000
batch_size = 256
beta_range = np.linspace(5, 0.05, 10)
# Deferred part
with tf.device('/device:CPU:0'):
    Lobs = tf.placeholder(dtype=tf.float32, name='Lobs', shape=[N, N])
    rho = tf.placeholder(dtype=tf.float32, name='rho', shape=[N, N])
    beta = tf.placeholder(dtype=tf.float32, name='beta', shape=())
    eta = tf.placeholder(dtype=tf.float32, name='eta', shape=())

    # Declare optimization variables initialization
    x0 = tf.random_uniform(shape=[batch_size, N],
                           minval=0.1, maxval=5, dtype=tf.float32)
    # Declare the optimization variables
    xi = tf.Variable(x0, name='xi', dtype=tf.float32, trainable=True)

    # sampling phase, create the random probabilities of sampling rij, with
    # size batch_size,N,N
    rij = tf.random_uniform([batch_size, N, N], minval=0.0, maxval=1.0)
    rij = (tf.linalg.transpose(rij) + rij) / 2.0
    # set matrix diagonal to zero, over all batches
    rij = tf.matrix_set_diag(rij, tf.constant([0.0], shape=[batch_size, N]))
    # Through einstein summation we perform outer product over all batch
    # examples (great example of einsum usage)
    xij = tf.einsum('ij,ik->ijk', xi, xi)
    # We define the edge picking probability as from the Undirected Binary
    # Configuration Model
    pij = xij / (1.0 + xij)  # explicit broadcasting of 1.0

    activation = 'sigmoid'
    slope = 50
    # Create the model adjacency matrix by sampling from pij with
    # probabilities rij
    if activation is 'sigmoid':
        # this is like setting < to 0 and > to 1, with shape [batch_size, N, N]
        Amodel = 1.0 / (1.0 + tf.exp(-slope * (-rij + pij)))
        #Amodel = tf.sigmoid(slope*(pij-rij))
    elif activation is 'tanh':
        # this is like setting < to 0 and > to 1, with shape [batch_size, N, N]
        Amodel = tf.tanh(slope * (pij - rij)) * 0.5 + 0.5
    elif activation is 'relu':
        Amodel = tf.nn.relu(tf.sign(pij - rij))
    elif activation is 'relaxedbernoulli':
        Amodel = tfp.distributions.RelaxedBernoulli(temperature=1, probs=pij).sample()
        Amodel = (Amodel + tf.linalg.transpose(Amodel)) / 2.0
        Amodel = tf.matrix_set_diag(Amodel, tf.constant([0.0], shape=[batch_size, N]))
    # Set the diagonal to zero
    Amodel = tf.matrix_set_diag(
        Amodel,  tf.constant([0.0], shape=[batch_size, N]))

    # Compute the model Laplacian
    Lmodel = tf.matrix_diag(tf.reduce_sum(Amodel, axis=1)) - Amodel
    # ELmodel = tf.matrix_diag(tf.reduce_sum(pij,axis=1)) - pij # this
    # estimate is better!

    # Define loss function
    # Model energy
    Lmodelrho = tf.multiply(rho, Lmodel, name='Lmrho')
    #ELmodelrho = tf.multiply(rho, ELmodel, name='Lmrho')

    # This is the ensemble average of <Trace[dot(rho,Lmodel)]>
    # using the fact that Trace[dot(rho,Lmodel)] = sum(rho.*Lmodel)
    # the mean is then taken on the batch dimension
    #Em = tf.reduce_mean(tf.reduce_sum(Lmodelrho, name='TrLmrho',axis=[1,2]))
    # but we can do better and put the exact expected value of <L>
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
    #rho = tf.exp(-beta*lo)
    #sigma = tf.exp(-beta*lm)/tf.reduce_sum()
    grads = tf.gradients(rel_entropy, xs=xi)
    # Define the optimizer, use the optimal learning rate as indicated by Soatto
    # https://arxiv.org/pdf/1710.11029.pdf
    #learning_rate = 2*batch_size / beta
    method = 'ADAM'
    if method is 'ADAM':
        global_step = tf.Variable(0.0, trainable=False)
        starter_learning_rate = eta
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step, nepochs*len(beta_range), 0.99,
                                                   staircase=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train = optimizer.minimize(rel_entropy, global_step=global_step)

    # Initialize the global variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        collect_dict = {'rel_entropy': [],
                        'gradients': [], 'sol': [], 'beta': []}

        # Prepare the execution of the computation graph
        sess.run(init)
        logging.info('==== Start model fitting ====')

        from drawnow import drawnow, figure
        # if global namespace, import plt.figure before drawnow.figure
        figure(figsize=(5, 10))
        # for b in tqdm(beta_range):
        for b in beta_range:
            brho = nq.compute_vonneuman_density(L=L, beta=b)
            feed_dict = {Lobs: L, beta: b, eta: 0.0001, rho: brho}
            for epoch in tqdm(range(nepochs), desc='beta=%g' % b):
                # for epoch in range(nepochs):
                LmodelTF = sess.run(Lmodel, feed_dict=feed_dict)
                if np.sum(np.isnan(LmodelTF)):
                    print('============== FOUND NAN =========')

                tfvals = sess.run(
                    [train, learning_rate, rel_entropy], feed_dict=feed_dict)
                collect_dict['rel_entropy'].append(tfvals[-1])
                print('\rLearning rate=', tfvals[1], end='')
                if (epoch % refresh_frames) == 0:
                    def draw_fig():
                        plt.subplot(4, 2, 1)
                        plt.imshow(A)
                        plt.title('Empirical network')
                        plt.grid(False)

                        plt.subplot(4, 2, 2)
                        A0 = sess.run(Amodel, feed_dict=feed_dict)[0]
                        ki = A.sum(axis=0)
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

                        # plt.subplot(4,2,5)
                        # import bct
                        # clu_obs = bct.clustering_coef_bu(A)
                        # for i in range(batch_size):
                        #     A0 = np.squeeze(sess.run(Amodel,feed_dict=feed_dict)[i,:,:])
                        #     clu_model = bct.clustering_coef_bu(A0)
                        #     plt.plot(clu_obs-clu_model,'.r')
                        # plt.plot(np.linspace(0,0.5,10),np.linspace(0,0.5,10),'k-')

                        plt.subplot(4, 2, 5)

                        # for i in range(batch_size):
                        A0 = np.squeeze(
                            sess.run(Amodel, feed_dict=feed_dict)[0, :, :])
                        k0 = np.sum(A0, axis=0)
                        plt.plot(range(N), ki - k0, '.r')
                        # plt.stem(np.squeeze(np.array(sess.run(grads,feed_dict=feed_dict)).mean(axis=1)),'-')
                        # plt.xlabel('i')
                        # plt.ylabel('g_i')
                        # plt.title('gradients')
                        # plt.grid(False)

                        plt.subplot(4, 2, 6)
                        plt.plot(collect_dict['rel_entropy'], '-')
                        plt.ylabel('<S(rho,sigma)>')
                        plt.xlabel('iteration')
                        plt.title('<S(rho,sigma)>')

                        plt.subplot(4, 2, 7)
                        plt.hist(sess.run(lm, feed_dict=feed_dict).flatten(), N, density=True, alpha=0.7)
                        plt.hist(lambda_obs, density=True, alpha=0.7)
                        plt.title('Laplacian eigs')
                        plt.xlabel('eigenvalues')
                        plt.ylabel('norm. frequency')
                        plt.legend(['Model', 'Obs'])

                        plt.subplot(4, 2, 8)
                        #Ri = np.random.random([batch_size,N,N])
                        #Ri += np.transpose(R, axes=[0,2,1])
                        #idx_diag = [ i+j*N*N for j in range(batch_size) for i in np.arange(0, N*N, N+1) ]
                        #Ri.ravel()[idx_diag] = 0
                        #ki = R.sum(axis=1)
                        #Lclassic = np.
                        for i in range(batch_size):
                            A0 = sess.run(Amodel, feed_dict=feed_dict)[i]
                            L0 = np.diag(np.sum(A0, axis=0)) - A0
                            plt.semilogx(np.logspace(-3, 3, 50), nq.batch_compute_vonneumann_entropy(L0, np.logspace(-3, 3, 50)), 'r-', alpha=0.5)
                        plt.semilogx(
                            np.logspace(-3, 3, 50), nq.batch_compute_vonneumann_entropy(L, np.logspace(-3, 3, 50)), 'b-')
                        # plt.legend(['sigma','rho'])
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
