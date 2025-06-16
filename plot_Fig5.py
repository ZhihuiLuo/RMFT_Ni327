# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from func import *
from H4band import *
from scipy.interpolate import LinearNDInterpolator
I = np.newaxis

amp = np.real

Ns = 2 # 2 spin for each orbital

Fkx = lambda x: 2*np.cos(x)
Fky = Fkx
def Gt(n,d):
    return np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d))
def GJ(n,d):
    assert np.all(n>=0), 'ERROR in n'
    if n==0: return 1
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.
def HJ(x,y,chi0x,chi0y,chi0z,dlt0x,dlt0y,dlt0z,J1,J2,J3,J4,J5,J6,JH=0,fc=float):
    m = np.zeros(np.shape(x)+(Nb*Ns,Nb*Ns),fc)
    
    Jinp = np.kron(s0, [[J4,J3],[J3,J1]])   # s0,sx are pauli matrix
    Japc = np.kron(sx, [[J2,J5],[J5,J6]]) + np.kron(s0, [[0,JH],[JH,0]])
    
    fkx,fky = Fkx(x)[...,np.newaxis,np.newaxis], Fky(y)[...,np.newaxis,np.newaxis]
    
    m[...,:Nb,:Nb] = -3./8*(chi0x*fkx+chi0y*fky)*Jinp -3./8*chi0z*Japc
    m[...,Nb:,Nb:] = -m[...,:Nb,:Nb]
    
    m[...,:Nb,Nb:] = -3./8*(dlt0x*fkx+dlt0y*fky)*Jinp -3./8*dlt0z*Japc
    
    if fc is float:
        m[...,Nb:,:Nb] = m[...,:Nb,Nb:].swapaxes(-1,-2)
    else:
        m[...,Nb:,:Nb] = m[...,:Nb,Nb:].swapaxes(-1,-2).conj()
    
    return m
def Hmf(km,n,pms,Js,JH,fc=float,Htb=Hk):  # Js is a tuple  (J,)
    # dn always is tuple like 
    d = 0 # docc
    # print('n=',n)
    Gtz2,Gtx2 = Gt(n[0]*Ns,d), Gt(n[1]*Ns,d)
    GJz2,GJx2 = GJ(n[0]*Ns,d), GJ(n[1]*Ns,d)
    
    # temporary !!!!!! ================================
    # if np.isnan(Gtz2): Gtz2 = 0
    # if np.isnan(Gtx2): Gtx2 = 0
    # if np.isnan(GJz2): GJz2 = 0
    # if np.isnan(GJx2): GJx2 = 0
    assert not np.any(np.isnan((Gtz2,Gtx2,GJz2,GJx2))), 'ERROR is G'
    # =================================================
    # print(Gtz2,Gtx2,GJz2,GJx2)
    one = np.ones([2,2])
    gt = np.kron(Gtz2*Gtz2*one,s00)+np.kron(Gtx2*Gtx2*one,s11)+np.kron(Gtx2*Gtz2*one,sx)
    # print(gt)
    
    gJz2,gJx2,gJzx = GJz2*GJz2, GJx2*GJx2, GJz2*GJx2
    Jsg = np.array(Js)*[gJx2,gJz2,gJzx,gJz2,gJzx,gJx2]
    
    m = HJ(*km,*pms,*Jsg,JH,fc=fc)
    m[...,:Nb,:Nb] += Htb(*km)*gt
    m[...,Nb:,Nb:] -= Htb(*-km)*gt
    return m

def getHk(km,n,pms,Js,mu,JH=0):
    if isinstance(pms[0].flat[0], complex): fc = complex
    else: fc = float
    hk = Hmf(km,n,pms,Js,JH,fc=fc)-np.diag(mu)
    return hk


# Mu2.npy simply store the chemical potentials for different dopping level of 
# the tight-binding Hamiltonian defined in H4band.py
# You can also calculate them by yourself
Mu2=np.load('Mu2.npy')

u = [[1,0,1,0],[0,1,0,1],[1,0,-1,0],[0,1,0,-1]]/np.sqrt(2)
U = np.kron(s0,u)

def H4b(hk):
    h = np.matmul(np.matmul(u.T,hk),u)
    h[...,:2,2:] = 0
    h[...,2:,:2] = 0
    return h

def H8b(hk):
    h0 = np.matmul(np.matmul(U.T,hk),U)
    h = np.zeros_like(hk)
    
    h[...,:4,:4] = h0[...,[[0],[1],[4],[5]], [[0,1,4,5]] ]
    h[...,4:,4:] = h0[...,[[2],[3],[6],[7]], [[2,3,6,7]] ]
    
    return h

def get_FS_Dlt(km,Dlt,xi,yi):
    Nk = km.shape[-1]
    kl = km.reshape([2,Nk**2])
    f = LinearNDInterpolator(kl.T,Dlt.flatten())
    dlt = f(xi,yi)
    return dlt


kl = (100,100); Nk = np.prod(kl)
km = kmesh(kl,close=True,xlim=(-np.pi,np.pi),ylim=(-np.pi,np.pi))
#%% =========================================================================
fig,ax = plt.subplots(2,2,figsize=(6,6),sharex=True,sharey=True)


vm,vx = -0.05,0.05
vm2,vx2 = -0.2,0.2
s = 4


hkm = Hk(*km)-np.diag(Mu2[0])
h4b = H4b(hkm)


vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])




#%%
fn = 'pms_p0_0_J0.09_0.18_0.npz'
dt = np.load(fn)
n = dt['n']; pms = dt['pms']; Js=dt['Js']; mu = dt['mu']
hkm = getHk(km,n,pms,Js,mu)

kmp = km[...,I,I]
p4 = pms[3]*np.cos(kmp[0])+pms[4]*np.cos(kmp[1])+pms[5]

p4b = np.matmul(np.matmul(u.T,p4),u)
pb = p4b[...,:2,:2]
pa = p4b[...,2:,2:]


pbr = np.matmul(np.matmul(eb.swapaxes(-1,-2).conj(),pb),eb)
par = np.matmul(np.matmul(ea.swapaxes(-1,-2).conj(),pa),ea)


#%%
plt.sca(ax[0,0])

plt.gca().set_aspect(1)


# plt.plot(x,y)
plt.rc('font',family='serif')


cmap = 'seismic'
# gamma pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm,vmax=vx)

# beta pocket
cs = plt.contour(km[0],km[1],amp(va[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(par[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm,vmax=vx)
    
# alpha pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,1]),levels=[0],alpha=0,linestyles='dashed')
for i in range(1):  # 1 branch
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,1,1]),x,y)
    plt.scatter(x,y,c=dlt,s=s, cmap=cmap,vmin=vm,vmax=vx)


plt.rc('text',usetex=True)
plt.xticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)
plt.yticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)



plt.text(0.9,-0.2,'$\\alpha$',fontsize=20)
plt.text(2.2,0.5,'$\\beta$',fontsize=20)
plt.text(2.3,2.3,'$\\gamma$',fontsize=20)


#%% =========================================================================
hkm = Hk(*km)-np.diag(Mu2[1])
h4b = H4b(hkm)


vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])


#%%
fn = 'pms_p0_-0.28_J0.09_0.18_0.npz'
dt = np.load(fn)
n = dt['n']; pms = dt['pms']; Js=dt['Js']; mu = dt['mu']
hkm = getHk(km,n,pms,Js,mu)

p4 = pms[3]*np.cos(kmp[0])+pms[4]*np.cos(kmp[1])+pms[5]

p4b = np.matmul(np.matmul(u.T,p4),u)
pb = p4b[...,:2,:2]
pa = p4b[...,2:,2:]


pbr = np.matmul(np.matmul(eb.swapaxes(-1,-2).conj(),pb),eb)
par = np.matmul(np.matmul(ea.swapaxes(-1,-2).conj(),pa),ea)


#%%
plt.sca(ax[0,1])

plt.gca().set_aspect(1)


# plt.plot(x,y)
plt.rc('font',family='serif')


cmap = 'seismic'
# gamma pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm2,vmax=vx2)

# beta pocket
cs = plt.contour(km[0],km[1],amp(va[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(par[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt,s=s, cmap=cmap,vmin=vm2,vmax=vx2)
    
# alpha pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,1]),levels=[0],alpha=0,linestyles='dashed')
for i in range(1):  # 1 branch
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,1,1]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm2,vmax=vx2)

# plt.colorbar()

plt.rc('text',usetex=True)
plt.xticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)
plt.yticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)



#%% =========================================================================
hkm = Hk(*km)-np.diag(Mu2[2])
h4b = H4b(hkm)


# vb = np.linalg.eigvalsh(h4b[:,:,:2,:2])
# va = np.linalg.eigvalsh(h4b[:,:,2:,2:])
vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])


#%%
fn = 'pms_p-0.15_0_J0.09_0.18_0.npz'
dt = np.load(fn)
n = dt['n']; pms = dt['pms']; Js=dt['Js']; mu = dt['mu']
hkm = getHk(km,n,pms,Js,mu)

p4 = pms[3]*np.cos(kmp[0])+pms[4]*np.cos(kmp[1])+pms[5]

p4b = np.matmul(np.matmul(u.T,p4),u)
pb = p4b[...,:2,:2]
pa = p4b[...,2:,2:]

pbr = np.matmul(np.matmul(eb.swapaxes(-1,-2).conj(),pb),eb)
par = np.matmul(np.matmul(ea.swapaxes(-1,-2).conj(),pa),ea)


#%%
plt.sca(ax[1,0])

plt.gca().set_aspect(1)


# plt.plot(x,y)
plt.rc('font',family='serif')


cmap = 'seismic'
# gamma pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm,vmax=vx)

# beta pocket
cs = plt.contour(km[0],km[1],amp(va[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(par[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm,vmax=vx)
    
# alpha pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,1]),levels=[0],alpha=0,linestyles='dashed')
for i in range(1):  # 1 branch
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,1,1]),x,y)
    im = plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm,vmax=vx)

# plt.colorbar()

plt.rc('text',usetex=True)
plt.xticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)
plt.yticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)



#%% =========================================================================
hkm = Hk(*km)-np.diag(Mu2[3])
h4b = H4b(hkm)

# vb = np.linalg.eigvalsh(h4b[:,:,:2,:2])
# va = np.linalg.eigvalsh(h4b[:,:,2:,2:])
vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])


#%%
fn = 'pms_p-0.15_-0.2_J0.09_0.18_0.npz'
dt = np.load(fn)
n = dt['n']; pms = dt['pms']; Js=dt['Js']; mu = dt['mu']
hkm = getHk(km,n,pms,Js,mu)

p4 = pms[3]*np.cos(kmp[0])+pms[4]*np.cos(kmp[1])+pms[5]

p4b = np.matmul(np.matmul(u.T,p4),u)
pb = p4b[...,:2,:2]
pa = p4b[...,2:,2:]

pbr = np.matmul(np.matmul(eb.swapaxes(-1,-2).conj(),pb),eb)
par = np.matmul(np.matmul(ea.swapaxes(-1,-2).conj(),pa),ea)


#%%
plt.sca(ax[1,1])

plt.gca().set_aspect(1)

plt.rc('font',family='serif')


cmap = 'seismic'


# beta pocket
cs = plt.contour(km[0],km[1],amp(va[:,:,0]),levels=[0],alpha=0)
for i in range(4):  # 4 branches
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(par[:,:,0,0]),x,y)
    plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm2,vmax=vx2)
    
# alpha pocket
cs = plt.contour(km[0],km[1],amp(vb[:,:,1]),levels=[0],alpha=0,linestyles='dashed')
for i in range(1):  # 1 branch
    x,y = cs.collections[0].get_paths()[i].vertices.T
    dlt = get_FS_Dlt(km,amp(pbr[:,:,1,1]),x,y)
    im2 = plt.scatter(x,y,c=dlt, s=s, cmap=cmap,vmin=vm2,vmax=vx2)



plt.rc('text',usetex=True)
plt.xticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)
plt.yticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)




ax[0,0].text(-4.7,2.6,'(a)',fontsize=17)    
ax[0,1].text(-4.2,2.6,'(b)',fontsize=17)  
ax[1,0].text(-4.7,2.6,'(c)',fontsize=17)  
ax[1,1].text(-4.2,2.6,'(d)',fontsize=17) 



cmap = 'jet'
cax = fig.add_axes([0.13, 0.04, 0.35, 0.02])
cbar = fig.colorbar(im, cax=cax,cmap=cmap,location='bottom')
cbar.ax.tick_params(labelsize=12)


cax = fig.add_axes([0.55, 0.04, 0.35, 0.02])
cbar = fig.colorbar(im2, cax=cax,cmap=cmap,location='bottom')
cbar.ax.tick_params(labelsize=12)

s = 50
ax[0,1].scatter(-3,3.5,marker='s',clip_on=False,c='k',s=s)
ax[0,0].scatter(-3,3.5,marker='o',clip_on=False,c='k',s=s)
ax[1,1].scatter(-3,3.5,marker='^',clip_on=False,c='k',s=s)
ax[1,0].scatter(-3,3.5,marker='v',clip_on=False,c='k',s=s)

fz = 17
ax[0,1].text(-1,3.4,'$d+is$',fontsize=fz)
ax[0,0].text(-0.7,3.4,'$s^{\pm}$',fontsize=fz)
ax[1,1].text(-0.4,3.4,'$d$',fontsize=fz)
ax[1,0].text(-1.3,3.4,'Normal',fontsize=fz)

ax[0,0].set_ylim(-pi,pi)


lw = 1.5
alpha = 1
c = 'grey'
z = 0
for i,a in enumerate(ax.flat):

    a.plot([pi,0],[0,pi],'--',c=c,lw=lw,alpha=alpha,zorder=z)
    a.plot([pi,0],[0,-pi],'--',c=c,lw=lw,alpha=alpha,zorder=z)
    a.plot([-pi,0],[0,pi],'--',c=c,lw=lw,alpha=alpha,zorder=z)
    a.plot([-pi,0],[0,-pi],'--',c=c,lw=lw,alpha=alpha,zorder=z)
    a.set_xlim(-pi,pi)
    a.set_ylim(-pi,pi)
    
    # a.text(-4.5,2.6,'(%s)'%chr(97+i),fontsize=15)    
    a.set_xticks([-pi,pi])
    a.set_yticks([-pi,pi])
    
    a.set_aspect(1)

plt.savefig('Fig5.pdf',bbox_inches='tight')
