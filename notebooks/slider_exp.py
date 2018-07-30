import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt, mpld3
from matplotlib.widgets import Slider, Button, RadioButtons
plt.style.use('ggplot')

import sys
sys.path.append('..')
import networkqit as nq
import networkx as nx

def signal(beta):
    y = np.exp(-beta*l)
    return y/np.sum(y)

def histogram(beta):
    y = signal(beta)
    freq, edges = np.histogram(y)
    return freq, edges


def von_neumann_entropy(beta):
    lrho = np.exp(-beta*l)
    Z = lrho.sum()
    return (np.log(Z) + beta * (l*lrho).sum()/Z)/np.log(len(A))

def von_neumann_density(L, beta):
    """ Get the von neumann density matrix :math:`\\frac{e^{-\\beta L}}{\\mathrm{Tr}[e^{-\\beta L}]}` """
    rho = expm(-beta*L)
    return rho / np.trace(rho)


fig, ax = plt.subplots(ncols=3,nrows=2,figsize=(16,16))

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.15, bottom=0.25)

#G = nx.planted_partition_graph(2,50,0.8,0.001)
#G = nq.ring_of_cliques(5,20)
#G = nx.karate_club_graph()
#A = nx.to_numpy_array(G)
#A = np.loadtxt('/home/carlo/workspace/communityalg/data/GroupAverage_rsfMRI_weighted.adj')
#G = nx.Graph(nx.from_numpy_array(np.loadtxt('/home/carlo/workspace/communityalg/data/GroupAverage_rsfMRI_weighted.adj')))
#G = list(nx.connected_component_subgraphs(nx.from_numpy_array(np.loadtxt('/home/carlo/workspace/communityalg/data/GroupAverage_rsfMRI_weighted.adj'))))[0]

#A = nx.to_numpy_array(G)
from scipy.io import loadmat
A = loadmat('/home/carlo/workspace/BCT/data_and_demos/Coactivation_matrix.mat')['Coactivation_matrix']
G = nx.from_numpy_array(A)

L = nq.graph_laplacian(A)
l,Q = np.linalg.eigh(L)
l = np.linalg.eigvalsh(L)

min_beta = -3
max_beta = 3
beta_0 = np.mean([max_beta,min_beta])
beta_range = np.logspace(min_beta, max_beta, 100)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax[0,0].plot(l, signal(beta_0), '.r-')
ax[0,0].set_xlabel('$\lambda(L)$')
ax[0,0].set_ylabel('$\lambda(\\rho)$')
ax[0,0].set_title('$\lambda(L) vs. \lambda(\\rho)$')

# the histogram of the data
[hist_lines] = ax[0,1].plot(histogram(beta_0)[1][:-1],histogram(beta_0)[0],drawstyle='steps')
#ax[0,1].fill_between(histogram(beta_0)[1],0,histogram(beta_0)[0])
ax[0,1].set_title('Histogram')
ax[0,1].set_xlabel('$\lambda(\\rho)$')
ax[0,1].set_ylabel('Frequency')

# the eigenvalues of rho
[rholines] = ax[0,2].plot(range(0,len(A)),signal(beta_0),'.r-',drawstyle='steps')
ax[0,2].set_title('eigs $\\rho$')
ax[0,2].set_ylabel('$\lambda(\\rho)$')
ax[0,2].set_xlabel('index')


# Spectral entropy plot
ax[1,0].semilogx(beta_range, [von_neumann_entropy(beta=b) for b in beta_range])
[entropy_lines] = ax[1,0].semilogx([beta_0],[von_neumann_entropy(beta=beta_0)],'ok')
ax[1,0].set_xlim([10**min_beta,10**max_beta])
ax[1,0].set_title('Entropy')
ax[1,0].set_ylabel('$S(\lambda(\\rho))$')
ax[1,0].set_xlabel('$\\beta$')

image = ax[1,1].imshow(von_neumann_density(L,beta_0), interpolation='none')
ax[1,1].grid(False)
ax[1,1].set_title('Density matrix')
# Define an axes area and draw a slider in it
exp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
exp_slider = Slider(exp_slider_ax, '$\\log_{10}(\\beta)$', min_beta, max_beta, valinit=beta_0)

#pos=nx.spring_layout(G,iterations=1000)
XYZ = loadmat('/home/carlo/workspace/BCT/data_and_demos/Coactivation_matrix.mat')['Coord']
pos = XYZ[:,0:2]

nx.draw_networkx_nodes (G=G, pos=pos,ax=ax[1,2], node_color='gray', vmin=0, vmax=1, node_size=1)
nx.draw_networkx_edges (G=G, pos=pos, ax=ax[1,2], edge_color='gray',edge_vmin=0,edge_vmax=1/len(A))
ax[1,2].axis('off')

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    v = signal(10**exp_slider.val)
    rho = von_neumann_density(L,10**(exp_slider.val))

    line.set_ydata(v)
    ax[0,0].set_ylim([0,np.max(v)*1.1])
    #hist_lines.set_ydata(histogram(exp_slider.val))
    #hist_lines.set_xdata(histogram_edges(exp_slider.val))

    rholines.set_ydata(v)
    ax[0,2].set_ylim([0,np.max(v)*1.1])
    entropy_lines.set_xdata([10**exp_slider.val])
    entropy_lines.set_ydata(von_neumann_entropy(10**exp_slider.val))

    nx.draw_networkx_nodes (G=G, pos=pos,ax=ax[1,2], node_color=rho.sum(axis=0), vmin=0, vmax=1, node_size=1)
    nx.draw_networkx_edges (G=G, pos=pos, ax=ax[1,2], edge_color=[rho[e[0]%(len(A)-1), e[1]%(len(A)-1)] for e in G.edges()],edge_vmin=0,edge_vmax=1/len(A),alpha=0.1)

    ax[0,1].set_ylim([-1,1.1*np.max(histogram(exp_slider.val)[0])])
    image.set_array(von_neumann_density(L,10**exp_slider.val))
    fig.canvas.draw_idle()

exp_slider.on_changed(sliders_on_changed)

#plt.tight_layout()
#plt.subplots_adjust()
plt.show()