import numpy as np
import matplotlib.pyplot as plt
from func import *
from H4band import *


def Gt(n,d):
    if n==0: return 1
    return (n>0)*1.*np.sqrt((n-2*d)/np.abs(n-n**2/2))*(np.sqrt(np.abs(1-n+d))+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
    # assert np.all(n<=1 and n>=0), 'ERROR in n'
    assert np.all(n>=0), 'ERROR in n'
    if n==0: return 1
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.

Nm = 220; Nm2 = Nm**2
km = kmesh([Nm,Nm],flatten=True)


fc = complex

if fc is float: ph = 1.
else: ph = np.exp(1j*0.1)

chi0x,chi0y,chi0z = np.ones([Nb,Nb],fc)*0.1, np.ones([Nb,Nb],fc)*0.11, np.ones([Nb,Nb],fc)*0.14
dlt0x,dlt0y,dlt0z = np.ones([Nb,Nb],fc)*0.1, -np.ones([Nb,Nb],fc)*0.1*ph, np.ones([Nb,Nb],fc)*0.14*ph
pms0 = chi0x,chi0y,chi0z,dlt0x,dlt0y,dlt0z



plt.rc('text', usetex=False)
plt.rc('font', family='serif')

fn = 'pams_figT_J0_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_J0.08_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_J0.09_0.18_0_0_0_0_p0_0_complex_T0_0.012_96.npz'


dt = np.load(fn)
Pms = dt['Pms']
Tr = dt['Tr']
N = dt['N'][0]

amp = np.abs

Dltd = Pms[:,3]-Pms[:,4]
Dlts = Pms[:,3]+Pms[:,4]
Dltsp = Pms[:,5]

Dltd1 = amp(Dltd[:,1,1])
Dlts1 = amp(Dlts[:,1,1])

Dltd2 = amp(Dltd[:,0,0])
Dlts2 = amp(Dlts[:,0,0])

Dltsp = amp(Dltsp[:,0,2])

if 1:
    d = 0
    Dltd1 *= Gt(N[1]*2,d)**2
    Dltd2 *= Gt(N[0]*2,d)**2
    Dlts1 *= Gt(N[1]*2,d)**2
    Dlts2 *= Gt(N[0]*2,d)**2
    Dltsp *= Gt(N[0]*2,d)**2
    
cd = 'purple'
cs = 'green'
lw = 2

Tsp = Tr/kB

plt.plot(Tsp,Dlts1,':',c=cs,label='$g_t^{||x}|\Delta_{s^\pm}^{||x}|$',lw=lw)
plt.plot(Tsp,Dlts2,'--',c=cs,label='$g_t^{||z}|\Delta_{s^\pm}^{||z}|$',lw=lw)
plt.plot(Tsp,Dltsp,'-',c=cs,label='$g_t^{\\bot z}|\Delta_{s^\pm}^{\\bot z}|$',lw=lw)

plt.plot(Tsp,Dltd1,':',c=cd,label='$g_t^{||x}|\Delta^{||x}_{d}|$',lw=lw)
plt.plot(Tsp,Dltd2,'--',c=cd,label='$g_t^{||z}|\Delta^{||z}_{d}|$',lw=lw)


plt.text(20,0.027,'p=0',fontsize=18)

plt.xlabel('T (K)',fontsize=15)
plt.gca().set_aspect(3.8e3)

plt.rc('text', usetex=True)
plt.legend(frameon=False,fontsize=16,loc=(0.57,0.3))
plt.ylabel('$g^\\nu_t|\Delta^\\nu_\\ell|$',fontsize=15)

plt.xlim(0,140)

# plt.gca().set_aspect(1.2e3)
plt.ylim(None,0.03)


plt.tick_params(axis='both',labelsize=12)

plt.savefig('Fig1.pdf',bbox_inches = 'tight')
