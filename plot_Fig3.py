import numpy as np
import matplotlib.pyplot as plt


def Gt(n,d):
    return np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d))
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

vmx = 0.04

# fout = 'pams_J0.1_0.2_0_0_0_0_complex_Nm220_-0.2_0.3_-0.15_0.3_101_91.npz'
fout = 'pams_J0.09_0.18_0_0_0_0_complex_Nm220_-0.3372_0.3_-0.1646_0.3_121_93.npz'

dt = np.load(fout)


J1,J2,J3 = dt['Js'][:3]

pxr,pzr = dt['pxr'], dt['pzr']
Pms = dt['Pms']

Dltx_J1 = Pms[:,:,3]
Dlty_J1 = Pms[:,:,4]
Dltz_J2 = Pms[:,:,5]

cl = 'k'

N = dt['N']
N[N>0.5] = 0.5
Ns,d = 2,0
gt = np.einsum('xzi,xzj->xzij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
Dltx_J1 *= gt
Dlty_J1 *= gt
Dltz_J2 *= gt


def phase(x):
    eps = 1e-4
    ph = np.imag(x)/np.abs(x)
    #ph[np.abs(x)<eps] = 0
    return ph

amp = np.abs
# amp = np.real

Dltd_inp = amp(Dltx_J1-Dlty_J1)
Dlts_inp = amp(Dltx_J1+Dlty_J1)
Dlts_apc = amp(Dltz_J2)


pzr, pxr = dt['pzr'],dt['pxr']
pmz,pmx = np.meshgrid(pzr,pxr)

slop = pzr[0]/pxr[0]

plt.rc('text', usetex=False)

from matplotlib.colors import LinearSegmentedColormap
fig, ax = plt.subplots()

cmap_name = 'dcm'
cmap_name = 'scm'
dcolors = [(0,'lightgrey'),(0.25,'green'),(0.75,'purple'),(1,'peru')]
scolors = [(0,1,0),(1,1,1)]

cm = LinearSegmentedColormap.from_list(cmap_name, dcolors,N=4)

eps = 0.0044
sz = (Dlts_inp[:,:,0,0]>eps)
szp = (Dlts_apc[:,:,0,2]>eps)
dz = (Dltd_inp[:,:,0,0]>eps)
dx = (Dltd_inp[:,:,1,1]>eps)

full = szp&dz&dx
normal = (~sz)&(~szp)&(~dz)&(~dx)
sis = (sz&szp)&(~full)
did = (dz&dx)&(~full)

# s1 = sz&(~sis)&(~full)
s2 = szp&(~sis)&(~full)
d1 = dz&(~did)&(~full)
d2 = dx&(~did)&(~full)

# sis = (Dlts_inp[:,:,0,0]>eps)*(Dlts_inp[:,:,0,2]>eps)*1.
# plt.pcolormesh(pmx,pmz,sis,edgecolor='face')

# did = (Dltd_inp[:,:,0,0]>eps)*(Dltd_inp[:,:,1,1]>eps)*1.
# did = (Dltd_inp[:,:,1,1]>eps)*1.
plt.pcolormesh(pmx,pmz,szp*0.4,edgecolor='face',cmap=cm,alpha=1,vmin=0,vmax=1)
plt.pcolormesh(pmx,pmz,dx*0.5,edgecolor='face',cmap=cm,alpha=dx,vmin=0,vmax=1)
plt.pcolormesh(pmx,pmz,(szp&dx)*1.,edgecolor='face',cmap=cm,alpha=(szp&dx)*1.,vmin=0,vmax=1)
# plt.colorbar()



ax.axhline(0,color='k',linestyle='--',alpha=0.3)
ax.axvline(0,color='k',linestyle='--',alpha=0.3)

ax.set_aspect(1)

plt.rc('text', usetex=True)
fz = 26
ax.text(0.13,-0.03,'$s^\pm$',fontsize=fz)
ax.text(0.06,0.18,'Normal',fontsize=fz)
ax.text(-0.28,0.16,'$d$',fontsize=fz)
ax.text(-0.335,-0.05,'$d+is$',fontsize=fz)


fs = 22
ax.text(0.,-0.23,'$p_x$',fontsize=fs)
ax.text(-0.42,0.05,'$p_z$',fontsize=fs)

ax.plot([pxr[0],pxr[-1]],[pxr[0]*slop,pxr[-1]*slop],'--',lw=1,c='w',alpha=0.5)
ax.scatter(0,0,s=50,c=cl,marker='o',zorder=4)
ax.scatter(pxr[0],pzr[0],s=50,c=cl,marker='D',clip_on=False,zorder=4)
# ax.scatter(px,pz,s=60,c=cl,marker='^',zorder=4)
# ax.scatter(0.37*rx,0.37*rz,s=100,c=cl,marker='*',zorder=4)

ax.scatter(-0.2,-0.15,s=60,c=cl,marker='^',zorder=4)
ax.scatter(0,-0.15,s=60,c=cl,marker='v',zorder=4)
ax.scatter(-0.28,0,s=60,c=cl,marker='s',zorder=4)

# ax.tick_params(axis='both', which='minor')
# plt.minorticks_on()
from matplotlib.ticker import AutoMinorLocator
ax.yaxis.set_minor_locator(AutoMinorLocator(2))

# ax.set_xlim(-0.2,-0.17)

# plt.savefig('Fig4.pdf',bbox_inches = 'tight')