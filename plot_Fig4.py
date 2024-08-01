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
fig, ax = plt.subplot_mosaic("bc;de",figsize=(4.4,3.5))


cmap = 'jet'


# fig,ax = plt.subplots(3,2,figsize=(6,7.2))


vmin,vmax = 0,0.08
# vmin,vmax = 0,0.2
ax['d'].pcolormesh(pmx,pmz,Dltd_inp[:,:,1,1],vmin=vmin,vmax=vmax,edgecolor='face',cmap=cmap)
ax['d'].axhline(0,color='k',linestyle='--',alpha=0.3)
ax['d'].axvline(0,color='k',linestyle='--',alpha=0.3)
ax['d'].set_aspect(1)

im = ax['e'].pcolormesh(pmx,pmz,Dltd_inp[:,:,0,0],vmin=vmin,vmax=vmax,edgecolor='face',cmap=cmap)
ax['e'].axhline(0,color='k',linestyle='--',alpha=0.3)
ax['e'].set_aspect(1)
ax['e'].axvline(0,color='k',linestyle='--',alpha=0.3)

cax = fig.add_axes([0.92, 0.135, 0.022, 0.32])
fig.colorbar(im, cax=cax,cmap=cmap,ticks=[0,0.02,0.04,0.06,0.08])
# fig.colorbar(im, cax=cax,cmap=cmap)


# cmap = 'scm'

vmin,vmax = 0,0.04
# vmin,vmax = 0,0.1
ax['c'].pcolormesh(pmx,pmz,Dlts_inp[:,:,0,0],vmin=vmin,vmax=vmax,edgecolor='face',cmap=cmap)
ax['c'].axhline(0,color='k',linestyle='--',alpha=0.3)
ax['c'].axvline(0,color='k',linestyle='--',alpha=0.3)
ax['c'].set_aspect(1)

im = ax['b'].pcolormesh(pmx,pmz,Dlts_apc[:,:,0,2],vmin=vmin,vmax=vmax,edgecolor='face',cmap=cmap)
# ax[0,0].title.set_text('apical s-wave')
ax['b'].axhline(0,color='k',linestyle='--',alpha=0.3)
ax['b'].axvline(0,color='k',linestyle='--',alpha=0.3)
ax['b'].set_aspect(1)

cax = fig.add_axes([0.92, 0.55, 0.022, 0.32])
fig.colorbar(im, cax=cax,cmap=cmap,ticks=[0,0.01,0.02,0.03,0.04])
# fig.colorbar(im, cax=cax,cmap=cmap)


# ax['a'].set_xlabel('$p_x$')


plt.sca(ax['b'])
# plt.xticks([-0.2,0.2],['',''])
plt.sca(ax['c'])
# plt.xticks([-0.2,0.2],['',''])
plt.yticks([0,0.2])
plt.sca(ax['e'])
plt.yticks([0,0.2])

# fig.tight_layout(pad=0.0)

slop = pzr[0]/pxr[0]
# ax['a'].plot([pxr[0],pxr[-1]],[pxr[0]*slop,pxr[-1]*slop],'--',lw=2,c='w')
# ax[0,1].plot([pxr[0],pxr[-1]],[pxr[0]*slop,pxr[-1]*slop],'--',lw=2,c='w')
# ax[1,0].plot([pxr[0],pxr[-1]],[pxr[0]*slop,pxr[-1]*slop],'--',lw=2,c='w')
# ax[1,1].plot([pxr[0],pxr[-1]],[pxr[0]*slop,pxr[-1]*slop],'--',lw=2,c='w')

plt.rc('text', usetex=True)
fz = 18
c = 'w'
ax['b'].text(-0.2,0.22,'$g_t^{\\bot z}|\Delta_{s^\pm}^{\\bot z}|$',fontsize=fz,c=c)
ax['c'].text(-0.2,0.22,'$g_t^{||z}|\Delta_{s^\pm}^{||z}|$',fontsize=fz,c=c)
ax['d'].text(-0.2,0.22,'$g_t^{||x}|\Delta_{d}^{||x}|$',fontsize=fz,c=c)
ax['e'].text(-0.2,0.2,'$g_t^{||z}|\Delta_{d}^{||z}|$',fontsize=fz,c=c)


fs = 16
ax['b'].text(-0.52,0.05,'$p_z$',fontsize=fs)
ax['d'].text(-0.52,0.05,'$p_z$',fontsize=fs)
ax['d'].text(-0.01,-0.3,'$p_x$',fontsize=fs)
ax['e'].text(-0.01,-0.3,'$p_x$',fontsize=fs)



for a in [ax['b'],ax['c'],ax['d'],ax['e']]:
    cl = 'w'
    s = 15
    # a.plot([pxr[0],pxr[-1]],[pxr[0]*slop,pxr[-1]*slop],'--',lw=1,c='w')
    a.scatter(0,0,s=s,c=cl,marker='o',zorder=4)
    # a.scatter(pxr[0],pzr[0],s=20,c=cl,marker='D',clip_on=False,zorder=4)
    # a.scatter(px,pz,s=30,c=cl,marker='^',zorder=4)
    # a.scatter(0.37*rx,0.37*rz,s=40,c=cl,marker='*',zorder=4)
    
    a.scatter(-0.2,-0.15,s=s,c=cl,marker='^',zorder=4)
    a.scatter(0,-0.15,s=s,c=cl,marker='v',zorder=4)
    a.scatter(-0.28,0,s=s,c=cl,marker='s',zorder=4)
    
plt.rc('text', usetex=False)
fs = 16
ax['b'].text(-0.53,0.27,'(a)',fontsize=fs)
ax['c'].text(-0.47,0.27,'(b)',fontsize=fs)
ax['d'].text(-0.53,0.27,'(c)',fontsize=fs)
ax['e'].text(-0.47,0.27,'(d)',fontsize=fs)
    
plt.savefig('Fig5.pdf',bbox_inches = 'tight')