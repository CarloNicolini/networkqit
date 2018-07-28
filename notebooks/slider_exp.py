import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
plt.style.use('ggplot')

import sys
sys.path.append('..')
import networkqit as nq
import networkx as nx

def signal(beta):
    y = np.exp(-(10**beta)*l)
    return y/np.sum(y)

def histogram(beta):
    y = signal(beta)
    freq, edges = np.histogram(y, len(A))
    return freq

def histogram_edges(beta):
    y = signal(beta)
    freq, edges = np.histogram(y, len(A))
    return edges

def von_neumann_entropy(beta):
    lrho = np.exp(-beta*l)
    Z = lrho.sum()
    return (np.log(Z) + beta * (l*lrho).sum()/Z)/np.log(len(A))

def vonneuman_density(L, beta):
    """ Get the von neumann density matrix :math:`\\frac{e^{-\\beta L}}{\\mathrm{Tr}[e^{-\\beta L}]}` """
    rho = expm(-beta*L)
    return rho / np.trace(rho)

fig, ax = plt.subplots(ncols=5,nrows=1,figsize=(36,12))

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.35)

G = nx.planted_partition_graph(2,50,0.8,0.001)
#G = nx.karate_club_graph()
A = nx.to_numpy_array(G)
L = nq.graph_laplacian(A)
l,Q = np.linalg.eigh(L)
l = np.linalg.eigvalsh(L)

min_beta = -3
max_beta = 3
beta_0 = np.mean([max_beta,min_beta])
beta_range = np.logspace(min_beta, max_beta, 100)

# Draw the initial plot
# The 'line' variable is used for modifying the line later
[line] = ax[0].plot(l, signal(beta_0), 'ro-')
ax[0].set_xlabel('$\lambda(L)$')
ax[0].set_ylabel('$\lambda(\\rho)$')
ax[0].set_title('Eigenvalues')

# the histogram of the data
[hist_lines] = ax[1].plot(histogram_edges(beta_0)[:-1],histogram(beta_0),drawstyle='steps')
ax[1].set_title('Histogram')
ax[1].set_xlabel('$\lambda(\\rho)$')
ax[1].set_ylabel('Frequency')

# Spectral entropy plot
ax[2].semilogx(beta_range, [von_neumann_entropy(beta=b) for b in beta_range])
[entropy_lines] = ax[2].semilogx([beta_0],[von_neumann_entropy(beta=beta_0)],'ok')
ax[2].set_xlim([10**min_beta,10**max_beta])
ax[2].set_title('Entropy')
ax[2].set_ylabel('$S(\lambda(\\rho))$')
ax[2].set_xlabel('$\\beta$')

image = ax[3].imshow(vonneuman_density(L,beta_0), interpolation='none')
ax[3].grid(False)
# Define an axes area and draw a slider in it
exp_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03])
exp_slider = Slider(exp_slider_ax, '$\\log_{10}(\\beta)$', min_beta, max_beta, valinit=beta_0)

nx.draw_networkx_nodes(G=G, pos=nx.spring_layout(G),ax=ax[4],node_color=Q[2,:], node_size=8)

# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    v = signal(exp_slider.val)
    line.set_ydata(v)
    ax[0].set_ylim([0,np.max(v)*1.1])
    hist_lines.set_ydata(histogram(exp_slider.val))
    entropy_lines.set_xdata([10**exp_slider.val])
    entropy_lines.set_ydata(von_neumann_entropy(10**exp_slider.val))
    ax[1].set_ylim([-1,1.1*np.max(histogram(exp_slider.val))])
    image.set_array(vonneuman_density(L,10**exp_slider.val))
    fig.canvas.draw_idle()
exp_slider.on_changed(sliders_on_changed)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    exp_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

plt.subplots_adjust()
#plt.tight_layout()
plt.show()
