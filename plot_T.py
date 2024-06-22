from RMFT_FUNC import *
import numpy as np
import matplotlib.pyplot as plt

Nm = 220; Nm2 = Nm**2
km = kmesh([Nm,Nm],flatten=True)


fc = complex

if fc is float: ph = 1.
else: ph = np.exp(1j*0.1)

chi0x,chi0y,chi0z = np.ones([Nb,Nb],fc)*0.1, np.ones([Nb,Nb],fc)*0.11, np.ones([Nb,Nb],fc)*0.14
dlt0x,dlt0y,dlt0z = np.ones([Nb,Nb],fc)*0.1, -np.ones([Nb,Nb],fc)*0.1*ph, np.ones([Nb,Nb],fc)*0.14*ph
pms0 = chi0x,chi0y,chi0z,dlt0x,dlt0y,dlt0z



#%% T
Js = 0.09,0.18,0.0,0,0,0
# p = 0,0,0,0
p = p0+p0
# phase = np.exp(1j*0.1)
# phase = 1

# pms = pms0
Pms,Mu,N = [],[],[]

eps = 1e-5

T0,Tmax,NT = 0,0.012,20
Tr = np.linspace(T0,Tmax,NT)
for iT,T in enumerate(Tr):
    print('iT= %d/%d ==============='%(iT,NT))
    
    pms,mu,n = scf(km,p,pms0,Js,eps=eps,fc=fc,T=T)
    Pms.append(pms); Mu.append(mu); N.append(n)
    
    pms0 = pms

Pms = np.array(Pms); Mu = np.array(Mu); N = np.array(N)

fc_map = {float:'float',complex:'complex'}
fn = 'pams_figT_J%g_%g_%g_%g_%g_%g'%Js+fc_map[fc]+'_T%g_%g_%d'%(T0,Tmax,NT)+'.npz'

np.savez(fn,Pms=Pms,Mu=Mu,N=N,eps=eps,Tr=Tr,Js=Js,p=p)
#%%
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

fn = 'pams_figT_J0_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_J0.08_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_J0.09_0.18_0_0_0_0_p0_0_complex_T0_0.012_96.npz'
# fn = 'pams_figT_J0.1_0.2_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
# fn = 'pms_p0_0_VxVxz0.8.npz'

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
plt.ylabel('$g^\\nu_t|\Delta^\\nu_\\alpha|$',fontsize=15)

plt.xlim(0,140)

# plt.gca().set_aspect(1.2e3)
plt.ylim(None,0.03)

# plt.plot([78,78],[-0.005,0.02],'--',c='r')
# plt.text(80,0.015,'$T_c^{exp,onset}=78\ K$',fontsize=15)

plt.tick_params(axis='both',labelsize=12)

plt.savefig('Fig1.pdf',bbox_inches = 'tight')
#%%
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

fn = 'pams_figT_J0_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_J0.08_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
# fn = 'pams_figT_J0.09_0.18_0_0_0_0_p0_0_complex_T0_0.012_96.npz'
# fn = 'pams_figT_J0.1_0.2_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
# fn = 'pms_p0_0_VxVxz0.8.npz'

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

plt.plot(Tsp,Dlts1,':',c=cs,label='$g_t^x|\Delta^{s\pm}_{//x}|$',lw=lw)
plt.plot(Tsp,Dlts2,'--',c=cs,label='$g_t^z|\Delta^{s\pm}_{//z}|$',lw=lw)
plt.plot(Tsp,Dltsp,'-',c=cs,label='$g_t^z|\Delta^{s\pm}_{\\bot z}|$',lw=lw)

plt.plot(Tsp,Dltd1,':',c=cd,label='$g_t^x|\Delta_{//x}^{d}|$',lw=lw)
plt.plot(Tsp,Dltd2,'--',c=cd,label='$g_t^z|\Delta_{// z}^{d}|$',lw=lw)


plt.text(20,0.024,'p=0',fontsize=18)

plt.xlabel('T (K)',fontsize=15)
plt.gca().set_aspect(3.8e3)

plt.rc('text', usetex=True)
plt.legend(frameon=False,fontsize=13,loc=(0.66,0.44))
plt.ylabel('$g_t|\Delta_\\alpha|$',fontsize=15)

plt.xlim(0,140)

# plt.gca().set_aspect(1.2e3)
plt.ylim(None,0.03)
plt.text(5,0.01,'$J_\\bot=2J_{//}=0.2$',fontsize=15)


# plt.plot([78,78],[-0.005,0.02],'--',c='r')
# plt.text(80,0.015,'$T_c^{exp,onset}=78\ K$',fontsize=15)

plt.tick_params(axis='both',labelsize=12)

plt.savefig('FigT2.pdf',bbox_inches = 'tight')

#%%
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# fn = 'pams_figT_J0.09_0.18_0_0_0_0_p0_0_complex_T0.00689386_0.00775559_96.npz'
fn = 'pams_figT_J0.09_0.18_0_0_0_0_p0_0_complex_T0.00689386_0.00861733_96.npz'
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
if 0:
    fnc = lambda x: 1/x
    Dltd1 = fnc(Dltd1)
    Dltd2 = fnc(Dltd2)
    Dlts1 = fnc(Dlts1)
    Dlts2 = fnc(Dlts2)
    Dltsp = fnc(Dltsp)

cd = 'purple'
cs = 'green'
lw = 2

Tsp = Tr/kB

plt.plot(Tsp,Dlts1,':',c=cs,label='$g_t^x|\Delta^{s\pm}_{//x}|$',lw=lw)
plt.plot(Tsp,Dlts2,'--',c=cs,label='$g_t^z|\Delta^{s\pm}_{//z}|$',lw=lw)
plt.plot(Tsp,Dltsp,'-',c=cs,label='$g_t^z|\Delta^{s\pm}_{\\bot z}|$',lw=lw)

plt.plot(Tsp,Dltd1,':',c=cd,label='$g_t^x|\Delta_{//x}^{d}|$',lw=lw)
plt.plot(Tsp,Dltd2,'--',c=cd,label='$g_t^z|\Delta_{// z}^{d}|$',lw=lw)


# plt.text(20,0.027,'p=0',fontsize=18)

plt.xlabel('T (K)',fontsize=15)

plt.rc('text', usetex=True)
plt.legend(frameon=False,fontsize=16,loc=(0.57,0.3))
plt.ylabel('$g_t|\Delta_\\alpha|$',fontsize=15)


# plt.xlim(0,140)

# plt.gca().set_aspect(1.2e3)
# plt.ylim(0,0.005)

# plt.plot([78,78],[-0.005,0.02],'--',c='r')
# plt.text(80,0.015,'$T_c^{exp,onset}=78\ K$',fontsize=15)

plt.tick_params(axis='both',labelsize=12)

#%%
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.xscale('log')
plt.plot(Tsp,Dlts1,':',c='g')
plt.plot(Tsp,Dlts2,'--',c='g')
plt.plot(Tsp,Dltsp,'-',c='g')
plt.ylim(0,0.0002)
plt.xlim(70,140)
plt.xlabel('lgT')
# plt.gca().set_aspect(1.3e3)
#%%
Js = 0.09,0.18,0.0,0,0,0
p = 0,0,0,0
# phase = np.exp(1j*0.1)
# phase = 1

# pms = pms0
Pms,Mu,N = [],[],[]

eps = 1e-5

T0,Tmax,NT = 80,90,4
Tr = np.linspace(T0,Tmax,NT)*kB
for iT,T in enumerate(Tr):
    print('iT= %d/%d ==============='%(iT,NT))
    
    pms,mu,n = scf(km,p,pms0,Js,eps=eps,fc=fc,T=T)
    Pms.append(pms); Mu.append(mu); N.append(n)
    
    pms0 = pms

Pms = np.array(Pms); Mu = np.array(Mu); N = np.array(N)

#%%
Gt = lambda n,d: np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d))

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
    Dltd1 *= Gt(N[1,0]*2,d)**2
    Dltd2 *= Gt(N[0,0]*2,d)**2
    Dlts1 *= Gt(N[1,0]*2,d)**2
    Dlts2 *= Gt(N[0,0]*2,d)**2
    Dltsp *= Gt(N[0,0]*2,d)**2
    

plt.plot(Tr/kB,Dlts1)
plt.plot(Tr/kB,Dltsp)
plt.plot(Tr/kB,Dlts2)

#%%
from RMFT_FUNC import Gt
from lzhpy.func import kB
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

fn = 'pams_figT_J0_0.16_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_J0.09_0.18_0.03_0_0_0_p0_0_complex_T0_0.012_128.npz'
fn = 'pams_figT_minimizeF_J0.09_0.18_0.03_0_0_0_p0_0_complex_T1e-05_0.012_96.npz'
# fn = 'pams_figT_J0.1_0.2_0_0_0_0_p0_0_complex_T0_0.012_128.npz'
# fn = 'pms_p0_0_VxVxz0.8.npz'

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

plt.plot(Tsp,Dlts1,':',c=cs,label='$g_t^x|\Delta^{s\pm}_{//x}|$',lw=lw)
plt.plot(Tsp,Dlts2,'--',c=cs,label='$g_t^z|\Delta^{s\pm}_{//z}|$',lw=lw)
plt.plot(Tsp,Dltsp,'-',c=cs,label='$g_t^z|\Delta^{s\pm}_{\\bot z}|$',lw=lw)

plt.plot(Tsp,Dltd1,':',c=cd,label='$g_t^x|\Delta_{//x}^{d}|$',lw=lw)
plt.plot(Tsp,Dltd2,'--',c=cd,label='$g_t^z|\Delta_{// z}^{d}|$',lw=lw)


plt.text(20,0.027,'p=0',fontsize=18)

plt.xlabel('T (K)',fontsize=15)
plt.gca().set_aspect(3.8e3)

# plt.rc('text', usetex=True)
plt.legend(frameon=False,fontsize=16,loc=(0.57,0.3))
plt.ylabel('$g_t|\Delta_\\alpha|$',fontsize=15)

plt.xlim(0,140)

# plt.gca().set_aspect(1.2e3)
plt.ylim(None,0.03)

# plt.plot([78,78],[-0.005,0.02],'--',c='r')
# plt.text(80,0.015,'$T_c^{exp,onset}=78\ K$',fontsize=15)

plt.tick_params(axis='both',labelsize=12)