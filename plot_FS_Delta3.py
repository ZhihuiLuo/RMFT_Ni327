# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from lzhpy.func import *
from lzhpy.eigenshuffle import *
from RMFT_FUNC import *
from scipy.interpolate import LinearNDInterpolator
I = np.newaxis

amp = np.real
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


# vb = np.linalg.eigvalsh(h4b[:,:,:2,:2])
# va = np.linalg.eigvalsh(h4b[:,:,2:,2:])
vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])


# vm,vx=-0.2,0.2
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\gamma$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\alpha$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\beta$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

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


# vm,vx=-0.02,0.02
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

#%%
# vm,vx=-0.05,0.05
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

# plt.colorbar()

plt.rc('text',usetex=True)
plt.xticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)
plt.yticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)

# plt.rc('text',usetex=False)
# plt.text(2.4,2.4,'-',fontsize=28)
# plt.text(-2.6,-2.6,'-',fontsize=25)
# plt.text(2.4,-2.6,'-',fontsize=26)
# plt.text(-2.6,2.4,'-',fontsize=25)
# plt.text(2.3,0.8,'+',fontsize=16)

plt.text(0.9,-0.2,'$\\alpha$',fontsize=20)
plt.text(2.2,0.5,'$\\beta$',fontsize=20)
plt.text(2.3,2.3,'$\\gamma$',fontsize=20)


#%% =========================================================================
hkm = Hk(*km)-np.diag(Mu2[1])
h4b = H4b(hkm)


# vb = np.linalg.eigvalsh(h4b[:,:,:2,:2])
# va = np.linalg.eigvalsh(h4b[:,:,2:,2:])
vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])

# vm,vx=-0.2,0.2
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\gamma$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\alpha$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\beta$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

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


# vm,vx=-0.02,0.02
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

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

# plt.rc('text',usetex=False)
# plt.text(2.4,2.4,'-',fontsize=28)
# plt.text(-2.6,-2.6,'-',fontsize=25)
# plt.text(2.4,-2.6,'-',fontsize=26)
# plt.text(-2.6,2.4,'-',fontsize=25)
# plt.text(2.3,0.8,'+',fontsize=16)

# plt.text(1,-0.2,'$\\alpha$',fontsize=20)
# plt.text(2.2,0.5,'$\\beta$',fontsize=20)
# plt.text(2.3,2.3,'$\\gamma$',fontsize=20)

#%% =========================================================================
hkm = Hk(*km)-np.diag(Mu2[2])
h4b = H4b(hkm)


# vb = np.linalg.eigvalsh(h4b[:,:,:2,:2])
# va = np.linalg.eigvalsh(h4b[:,:,2:,2:])
vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])

# vm,vx=-0.2,0.2
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\gamma$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\alpha$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\beta$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

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


# vm,vx=-0.02,0.02
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

#%%
# vm,vx=-0.05,0.05
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

# plt.rc('text',usetex=False)
# plt.text(2.4,2.4,'-',fontsize=28)
# plt.text(-2.6,-2.6,'-',fontsize=25)
# plt.text(2.4,-2.6,'-',fontsize=26)
# plt.text(-2.6,2.4,'-',fontsize=25)
# plt.text(2.3,0.8,'+',fontsize=16)

# plt.text(1,-0.2,'$\\alpha$',fontsize=20)
# plt.text(2.2,0.5,'$\\beta$',fontsize=20)
# plt.text(2.3,2.3,'$\\gamma$',fontsize=20)



#%% =========================================================================
hkm = Hk(*km)-np.diag(Mu2[3])
h4b = H4b(hkm)

# vb = np.linalg.eigvalsh(h4b[:,:,:2,:2])
# va = np.linalg.eigvalsh(h4b[:,:,2:,2:])
vb,eb = np.linalg.eigh(h4b[:,:,:2,:2])
va,ea = np.linalg.eigh(h4b[:,:,2:,2:])


# vm,vx=-0.2,0.2
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\gamma$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(vb[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\alpha$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.text(0,0,'$\\beta$',fontsize=20,c='w')
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(va[:,:,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

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


# vm,vx=-0.03,0.03
# plt.subplot(221)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(222)
# plt.pcolormesh(km[0],km[1],amp(pbr[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(223)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,0,0]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()
# plt.subplot(224)
# plt.pcolormesh(km[0],km[1],amp(par[:,:,1,1]),edgecolor='face',cmap='seismic',vmin=vm,vmax=vx)
# plt.colorbar()

#%%
plt.sca(ax[1,1])

plt.gca().set_aspect(1)


# plt.plot(x,y)
plt.rc('font',family='serif')


cmap = 'seismic'
# gamma pocket
# cs = plt.contour(km[0],km[1],amp(vb[:,:,0]),levels=[0],alpha=0)
# for i in range(1):  # 4 branches
#     x,y = cs.collections[0].get_paths()[i].vertices.T
#     dlt = get_FS_Dlt(km,amp(pbr[:,:,0,0]),x,y)
#     plt.scatter(x,y,c=dlt, cmap=cmap,vmin=vm,vmax=vx)

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

# plt.colorbar()

plt.rc('text',usetex=True)
plt.xticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)
plt.yticks([-np.pi,np.pi],['$-\pi$','$\pi$'],fontsize=16)

# plt.rc('text',usetex=False)
# plt.text(2.4,2.4,'-',fontsize=28)
# plt.text(-2.6,-2.6,'-',fontsize=25)
# plt.text(2.4,-2.6,'-',fontsize=26)
# plt.text(-2.6,2.4,'-',fontsize=25)
# plt.text(2.3,0.8,'+',fontsize=16)

# plt.text(1,-0.2,'$\\alpha$',fontsize=20)
# plt.text(2.2,0.5,'$\\beta$',fontsize=20)
# plt.text(2.3,2.3,'$\\gamma$',fontsize=20)





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

plt.savefig('Fig_FS_Delta3.pdf',bbox_inches='tight')