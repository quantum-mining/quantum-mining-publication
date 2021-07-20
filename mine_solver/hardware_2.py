import itertools
from utils   import *
from problem import OpenPitMiningProblem
from config  import Configuration
from result  import Result
from qiskit  import *

from   matplotlib import rc

rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

color_red   = '#DC343B'
color_blue  = '#0079BF'
color_green = '#B9D146'

def get_pdf(parameters,VF):
    circ = VF.construct_circuit(parameters)
    res  = qiskit.execute(circ,backend=Aer.get_backend('statevector_simulator')).result()
    prob = np.abs(res.get_statevector(circ))**2
    return prob

def DIST(V,W):
    return -np.log(np.sqrt(np.sum(V*W)))

config_filename = '../configs/config1.json'
configuration = Configuration(config_filename)
problem = OpenPitMiningProblem(configuration.get_problem_options())

VF = OpenPit_VarForm(num_qubits=4,num_layers=1,problem=problem,num_params=7)
p0 = [1.32626629,1.93758699,1.44853228,3.14159265,1.81532101,1.20400523,1.69305878]; p0 = get_pdf(p0,VF) # statevector
p1 = [2.42678417,2.00978447,0.51443224,3.12972106,1.12478717,1.13880863,2.44631943]; p1 = get_pdf(p1,VF) # hardware, raw
p2 = [1.10409849,2.13621200,1.18366335,3.20173417,1.96937087,0.98615057,1.86166442]; p2 = get_pdf(p2,VF) # hardware, em

for mu,idx in enumerate(itertools.product([0,1],repeat=4)):
    print(idx[::-1],p0[mu],p1[mu],p2[mu])

print(DIST(p0,p1))
print(DIST(p0,p2))
print(DIST(p1,p2))

import matplotlib.pyplot as plt
from   matplotlib import rc

L       = 5.0
fig,ax  = plt.subplots(1,1,figsize=(L,0.66*L))
plt.subplots_adjust(wspace=0,hspace=0)
X   = np.arange(16)
ax.bar(X - 0.25, p1, color = color_red,   width = 0.25, label = 'hardware, raw')
ax.bar(X + 0.00, p2, color = color_blue,  width = 0.25, label = 'hardware, em')
ax.bar(X + 0.25, p0, color = color_green, width = 0.25, label = 'exact')
ax.legend()
ax.set_yscale('log')
ax.set_xlabel('configuration')
ax.set_ylabel('probability')
ax.set_xlim([-0.5,15.5])
ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
ax.set_xticklabels(['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111'],rotation=45)
ax.set_ylim([7e-7,1+3e-1])
ax.set_yticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
ax.set_yticklabels([r'$10^{%d}$'%x for x in [-6,-5,-4,-3,-2,-1,0]])
fig.savefig('hardware_2.eps',format='eps',bbox_inches='tight')

