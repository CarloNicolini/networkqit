import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

import matplotlib as mpl
mpl.rcParams.update({
    'font.family':'sans-serif',
    'font.sans-serif':['Helvetica'],
    'svg.fonttype':'none'
    })
mpl.rc('text', usetex=False)

plt.figure(figsize=(10,5))
t = 1.5
z = np.arange(0,10)
plt.plot(z, (z>0) + z,'o--')
z = np.arange(0,10,0.01)
plt.plot(z, (z>=t) + z*(z>=t),'-.r')
plt.yticks(np.arange(0,10))
plt.xticks(np.arange(0,10))
plt.ylabel('$H$')
plt.xlabel('$w_{ij}$')
plt.legend(['UECM3 $H=\\sum_{i<j} \\alpha_{ij}\\Theta(w_{ij}) + \\beta_{ij}w_{ij}$','CWTECM $H=\\sum_{i<j} \\alpha_{ij}\\Theta(w_{ij}-t) + \\beta_{ij} w_{ij} \\Theta(w_{ij}-t)$'])
plt.title('Discrete vs continuos enhanced configuration model\n threshold=1.5')
plt.grid()
plt.savefig('discrete_vs_enhanced_cm.pdf',bbox_inches='tight')
plt.show()
