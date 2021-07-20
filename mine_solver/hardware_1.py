import numpy as np
import matplotlib.pyplot as plt
from   matplotlib import rc

color_red  = '#DC343B'
color_pink = '#f3babc'
color_blue = '#0079BF'
color_cyan = '#96d8ff'

theta_sv    = np.array([1.32626629,1.93758699,1.44853228,3.14159265,1.81532101,1.20400523,1.69305878])
theta_hw    = np.array([2.42678417,2.00978447,0.51443224,3.12972106,1.12478717,1.13880863,2.44631943])
theta_hw_em = np.array([1.10409849,2.136212  ,1.18366335,3.20173417,1.96937087,0.98615057,1.86166442])

def BAND(pan,m,s,c1,c2):
    pan.fill_between(x=[500,2000],y1=m-s,y2=m+s, color=c1)

def fill_panel(pan,xlabel,xlim,xticks,xticklabels,ylabel,ylim,yticks,yticklabels,p=20.0,q=10.0):
    x0,x1 = xlim
    xlim  = [x0-(x1-x0)/p,x1+(x1-x0)/p]
    pan.set_xlabel(xlabel)
    pan.set_xlim(xlim)
    pan.set_xticks(xticks)
    pan.set_xticklabels(xticklabels)
    pan.set_ylabel(ylabel)
    y0,y1 = ylim
    ylim  = [y0-(y1-y0)/q,y1+(y1-y0)/q]
    pan.set_ylim(ylim)
    pan.set_yticks(yticks)
    pan.set_yticklabels(yticklabels)
    pan.tick_params(direction='in',which='both')

rc('font',**{'family':'serif','serif':['cmu serif'],'size':12})
rc('text', usetex=True)

L = 5.0
fig,pan = plt.subplots(1,1,figsize=(L,0.66*L))
plt.subplots_adjust(wspace=0,hspace=0)

f          = open('solver_hardware_SPSA.out','r')
f          = f.readlines()
f          = [l.split() for l in f if 'Evaluation' in l]
y_hardware = [float(f[i][len(f[i])-1]) for i in range(len(f))]
y_hardware = -np.array(y_hardware[:2000])

f             = open('solver_hardware_em_SPSA.out','r')
f             = f.readlines()
f             = [l.split() for l in f if 'Evaluation' in l]
y_hardware_em = [float(f[i][len(f[i])-1]) for i in range(len(f))]
y_hardware_em = -np.array(y_hardware_em[:2000])

m,s = np.mean(y_hardware[500:]),np.std(y_hardware[500:])
print("HW, raw = ",m,s,s/np.sqrt(1500))
print("raw/SV angle difference max,ave = ",np.abs(theta_sv-theta_hw).max(),np.mean(np.abs(theta_sv-theta_hw)))
BAND(pan,m,s,color_pink,color_red)
m,s = np.mean(y_hardware_em[500:]),np.std(y_hardware_em[500:])
print("HW, mit = ",m,s,s/np.sqrt(1500))
print("mit/SV angle difference max,ave = ",np.abs(theta_sv-theta_hw_em).max(),np.mean(np.abs(theta_sv-theta_hw_em)))
BAND(pan,m,s,color_cyan,color_blue)

pan.plot(range(len(y_hardware)),   y_hardware,   marker='',color=color_red, linestyle='--',label='raw')
pan.plot(range(len(y_hardware_em)),y_hardware_em,marker='',color=color_blue,linestyle='-', label='mitigated')
pan.axhline(5,c='black',ls='-.',lw=0.75,label='exact')
pan.legend()

fill_panel(pan,'cost function evaluation',[0,2000],[0,500,1000,1500,2000],['0','500','1000','1500','2000'],
               'cost function',[1,5],[1,2,3,4,5],['1','2','3','4','5'],
               p=20.0,q=10.0)

fig.savefig('hardware_1.eps',format='eps',bbox_inches='tight')

