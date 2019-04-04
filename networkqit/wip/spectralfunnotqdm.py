from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import with_statement

import tensorflow as tf
from tensorflow.python.client import timeline

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import networkqit as nq
from scipy.linalg import expm, eigvalsh
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)

plt.style.use('ggplot')

tf.reset_default_graph()

# Imperative part
#G = nx.karate_club_graph()
G = nq.ring_of_cliques(10,10)
N = len(G.nodes())
A = nx.to_numpy_array(G)

def benchmark():
    ers = np.array([[250, 50], [50, 250]])
    ers = np.kron(ers,np.array([[1,0.8],[0.8,0.5]])*2)
    nr = [32, 32]*2
    return nq.sbm(ers,nr)
A = benchmark()
N = len(A)
m = A.sum()
L = np.diag(A.sum(axis=0)) - A

#beta_range = np.logspace(0.1,-1,2)
beta_range = np.linspace(1,0.1,10)
nepochs = int(5E4)
batch_size = 1024

lambda_obs = eigvalsh(L)

# Deferred part
with tf.device('/device:CPU:0'):

    Lobs = tf.placeholder(dtype=tf.float32,name='Lobs', shape=[N,N])
    rho = tf.placeholder(dtype=tf.float32,name='rho', shape=[N,N])
    beta = tf.placeholder(dtype=tf.float32,name='beta',shape=())
    eta = tf.placeholder(dtype=tf.float32,name='eta',shape=())
    
    # Declare optimization variables initialization
    x0 = tf.random_uniform(shape=[batch_size,N], minval=0.1, maxval=5, dtype=tf.float32)
    # Declare the optimization variables
    xi = tf.Variable(x0, name='xi', dtype=tf.float32, trainable=True)
    
    # sampling phase, create the random probabilities of sampling rij, with size batch_size,N,N
    rij = tf.random_uniform([batch_size,N,N], minval=0.0, maxval=1.0)
    rij = (tf.linalg.transpose(rij) + rij) / 2.0
    # set matrix diagonal to zero, over all batches
    rij = tf.matrix_set_diag(rij, tf.constant([0.0], shape=[batch_size,N]))
    # Through einstein summation we perform outer product over all batch examples (great example of einsum usage)
    xij = tf.einsum('ij,ik->ijk', xi, xi)
    # We define the edge picking probability as from the Undirected Binary Configuration Model
    pij = xij / (1.0 + xij) # explicit broadcasting of 1.0

    activation = 'tanh'
    slope = 500
    # Create the model adjacency matrix by sampling from pij with probabilities rij
    if activation is 'sigmoid':
        Amodel = 1.0 / (1.0 + tf.exp(slope*(rij-pij))) # this is like setting < to 0 and > to 1, with shape [batch_size, N, N]
    elif activation is 'tanh':
        Amodel = tf.tanh(slope*(pij-rij))*0.5 + 0.5 # this is like setting < to 0 and > to 1, with shape [batch_size, N, N]

    # Set the diagonal to zero
    Amodel = tf.matrix_set_diag(Amodel,  tf.constant([0.0],shape=[batch_size,N]))
    
    # Compute the model Laplacian
    Lmodel = tf.matrix_diag(tf.reduce_sum(Amodel,axis=1)) - Amodel
    ELmodel = tf.matrix_diag(tf.reduce_sum(pij,axis=1)) - pij # this estimate is better!
    
    # Define loss function
    
    # Model energy
    Lmodelrho = tf.multiply(rho, Lmodel, name='Lmrho')
    ELmodelrho = tf.multiply(rho, ELmodel, name='Lmrho')
    # This is the ensemble average of <Trace[dot(rho,Lmodel)]>
    # using the fact that Trace[dot(rho,Lmodel)] = sum(rho.*Lmodel)
    # the mean is then taken on the batch dimension
    #Em = tf.reduce_mean(tf.reduce_sum(Lmodelrho, name='TrLmrho',axis=[1,2]))
    # but we can do better and put the exact expected value of <L>
    Em = tf.reduce_mean(tf.reduce_sum(Lmodelrho, name='TrLmrho',axis=[1,2]))

    # Observation energy (observation eigenvalues don't need to be computed)
    #Lobsrho = tf.multiply(Lobs, rho,name='Lmrho')
    # here we just make a sum, because we miss a dimension (it's N x N)
    #Eo = tf.reduce_sum(Lobsrho, name='TrLmrho')

    # Model laplacian eigenvalues, shape is [batch_size,N]
    lm = tf.linalg.eigvalsh(Lmodel, name='lambda_model')
    # Observation laplacian eigenvalues, shape is [N,1]
    #lo = tf.linalg.eigvalsh(Lobs, name='lambda_obs')
    # Model free energy, reduce logsumexp on last dimension, then average over the ensemble
    Fm = - tf.reduce_mean(tf.reduce_logsumexp(-beta*lm,name='Fm',axis=1)) / beta
    # Observation free energy
    #Fo = - tf.reduce_logsumexp(-beta*lo,name='Fo') / beta
    # Loglikelihood and relative entropy to train
    #loglike = beta*(-Fm + Em)
    #entropy = beta*(-Fo + Eo)
    #rel_entropy = loglike - entropy
    loss = beta*Em - Fm
    
    # Define the optimizer, use the optimal learning rate as indicated by Soatto
    # https://arxiv.org/pdf/1710.11029.pdf
    #learning_rate = 2*batch_size / beta
    method = 'BFGS'
    if method is 'ADAM':
        optimizer = tf.train.AdamOptimizer(learning_rate = eta)
        train = optimizer.minimize(loss)
    
    if method is 'BFGS':
        optimizer = tf.train.
    if method is 'SGLD':
        import tensorflow_probability as tfp
        starter_learning_rate = 0.1
        end_learning_rate = 0.001
        decay_steps = 200
        global_step = tf.Variable(0, trainable=True)
        learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps, end_learning_rate, power=1.)
    
        # # Set up the optimizer
        optimizer_kernel = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=eta, preconditioner_decay_rate=0.99)
        train = optimizer_kernel.minimize(loss)

    
    # Initialize the global variables
    init = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        collect_dict = {'rel_entropy':[], 'gradients':[], 'sol':[], 'beta':[]}

        # Prepare the execution of the computation graph
        sess.run(init)
        #logging.info('==== Start model fitting ====')

        from drawnow import drawnow, figure
        # if global namespace, import plt.figure before drawnow.figure

        figure(figsize=(12,48))

        b = beta_range[0]
        brho = expm(-b*L)/np.trace(expm(-b*L))
        feed_dict = { Lobs:L, beta:b, eta: 0.001, rho: brho }
        #for b in beta_range:
        for epoch in range(nepochs):
            tmp = sess.run(train, feed_dict = feed_dict)
        logging.info('==== Done model fitting ====')
    
    #plt.show()
    #for i,b in enumerate(beta_range):
    #    plt.plot(np.linspace(i*nepochs,(i+1)*nepochs,nepochs),collect_dict['rel_entropy'][i*nepochs:(i+1)*nepochs])
    #plt.title('Relative entropy')
    #plt.xlabel('Iteration')

    #plt.show()
