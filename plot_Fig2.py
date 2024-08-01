import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from func import *
from H4band import *

def Gt(n,d):
    if np.any(n>1): 
        print('Warning: n '+str(n))
        n = 1

    return (n>0)*1.*np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
    assert np.all(n>=0), 'ERROR in n'
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.


fout = 'pams_fig1_alpha0.671965_J0.09_0.18_0_0_0_0_complex_Nm220_-0.501831_0.6_192.npz'
dt = np.load(fout)


J1,J2,J3 = dt['Js'][:3]

pr = dt['pr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]

# print('n0= ',n0)
alpha = dt['alpha']


N = dt['N']
N[N>0.5] = 0.5
Ns = 2.; d = 0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt


Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z



print(pr)

amp = np.abs
# amp = np.real

Dltd = amp(Dltd_inp )
Dlts = amp(Dlts_inp )
Dltsp = amp(Dlts_apc )



plt.rc('text', usetex=False)
plt.rc('font', family='serif')

cd = 'purple'
cs = 'green'
lw = 2

#fig,ax = plt.subplots(2,3)
plt.plot(pr, Dlts[:,1,1],':',c=cs,label='$g_t^{||x}|\Delta^{||x}_{s^\pm}|$',lw=lw,clip_on=False)
plt.plot(pr, Dlts[:,0,0],'--',c=cs,label='$g_t^{||z}|\Delta^{||z}_{s^\pm}|$',lw=lw,clip_on=False)
plt.plot(pr, Dltsp[:,0,2],ls='-',c=cs,label='$g_t^{\\bot z}|\Delta^{\\bot z}_{s^\pm}|$',lw=lw,clip_on=False)

plt.plot(pr, Dltd[:,1,1],':',c=cd,label='$g_t^{||x}|\Delta_{d}^{||x}|$',lw=lw,clip_on=False)
plt.plot(pr, Dltd[:,0,0],ls='--',c=cd,label='$g_t^{||z}|\Delta^{||z}_{d}|$',lw=lw,clip_on=False)

#plt.plot(pr, Dltd[:,0,1],'-.',c='r',label='$\Delta_d\ d_{x^2}-d_{z^2}$')
#plt.plot(pr, Dlts[:,0,1],'-.',c='b',label='$\Delta_d\ d_{x^2}-d_{z^2}$')
# plt.legend(frameon=False,fontsize=13,ncol=2,columnspacing=0.6,loc=(0.22,0.65))


plt.xlabel('p',fontsize=16)

# plt.annotate('', xy=(0,0), xytext=(0, 0.03),arrowprops=dict(ls='-', color='k',lw=0.1))
# plt.text(-0.05,0.03,'p=0',fontsize=13)

# plt.plot([pr[0],pr[0]],[-0., 0.03],'--',c='r',lw=4,zorder=10)
# plt.annotate('', xy=(pr[0],0), xytext=(pr[0], 0.02),arrowprops=dict(ls='-', color='r',lw=0.1))
# plt.text(-0.47,0.02,'half-filling',fontsize=13)

pp = 0.37
# plt.annotate('', xy=(pp,0), xytext=(pp, 0.005),arrowprops=dict(ls='-', color='r',lw=0.1))


plt.gca().tick_params(top=False)

# plt.text(-0.4,0.025,'d+is',fontsize=13)
# plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

cl = 'k'
# plt.axvline(pr[0],c=cl,ls='--',lw=1)
# plt.axvline(-0.39,c=cl,ls='--',ymax=0.94,lw=1)
# plt.axvline(0,c=cl,ls='--',ymax=0.45,lw=1)
# plt.axvline(pp,c=cl,ls='--',ymax=0.17,lw=1)


plt.scatter(-0.5,0,marker='D',c=cl,s=40,zorder=5,clip_on=False)
# plt.scatter(-0.39,0,marker='^',c=cl,s=40,zorder=5)
plt.scatter(0,0,marker='o',c=cl,s=40,zorder=5,clip_on=False)
# plt.scatter(pp,0,marker='*',c=cl,s=70,zorder=5)

# plt.text(-0.05,0.028,'p=0',fontsize=13)
# plt.text(-0.05,0.028,'p=0',fontsize=13)

plt.rc('text', usetex=True)
# plt.rc('mathtext.fontset',)
plt.ylabel('$g_t^{\\nu}|\Delta^\\nu_\\ell|$',fontsize=16)
plt.legend(frameon=False,fontsize=14,ncol=2,columnspacing=0.6,loc=(0.24,0.62))

# plt.text(0.16,0.024,'$J_1=2J_2=%g$'%J2,fontsize=15)
# plt.text(-0.22,0.06,'$J_2=%g$'%J1,fontsize=16)
# plt.text(-0.22,0.045,'$J_3=%g$'%J3,fontsize=16)





plt.gca().xaxis.set_minor_locator(MultipleLocator(0.1))

plt.gca().set_aspect(15)

plt.xlim(pr[0],pr[-1])

plt.ylim(0.00,0.06)

plt.text(0.-0.15,0.012,'$s^\pm$',fontsize=18)
plt.text(-0.45,0.012,'$d+is$',fontsize=16)
plt.axvline(-0.26,c='grey',ls='--',ymax=0.4,lw=1)

plt.text(0,0.004,'PC',fontsize=12)
plt.text(-0.57,-0.006,'HF',fontsize=12)

# plt.fill_between([-0.256,0.6],[0.06,0.06],y2=0,color='green',alpha=0.2,zorder=-1)

plt.savefig('Fig2.pdf',bbox_inches = 'tight')
plt.show()