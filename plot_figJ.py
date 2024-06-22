import numpy as np
import matplotlib.pyplot as plt
# from RMFT_FUNC import *
plt.rc('font', family='serif')
plt.rc('text', usetex=False)

def Gt(n,d):
    # assert np.all(n<=1 and n>=0), print(n); 'ERROR in n'
    # assert np.all((n>=0)*(n<=1)), 'ERROR in n'
    if np.any(n>1): 
        print('Warning: n '+str(n))
        n = 1
    # if n==0: return 1
    return (n>0)*1.*np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
    # assert np.all(n<=1 and n>=0), 'ERROR in n'
    assert np.all(n>=0), 'ERROR in n'
    # if n==0: return 1
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.

# -----------------------------------------------------
fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH0.npz'
# fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH-0.5.npz'


dt = np.load(fn)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z

amp = np.abs

Dltd = amp(Dltd_inp )
Dlts = amp(Dlts_inp )
Dltsp = amp(Dlts_apc )


dj = Jr[:,0]/Jr[:,1]

#% -----------------------------------------------------
# fn2 = 'pams_figJ_p-0.15_0_complex_Nm220_0_2_96_JH0.npz'
# fn2 = 'pams_figJ_p0_-0.1_complex_Nm220_0_2_96_JH0.npz'
fn2 = 'pams_figJ_p0_-0.28_complex_Nm220_0_2_96_JH0.npz'

dt = np.load(fn2)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z


Dltd2 = amp(Dltd_inp )
Dlts2 = amp(Dlts_inp )
Dltsp2 = amp(Dlts_apc )


# %---------------------------------------------------------
plt.plot(dj,Dltsp[:,0,2],c='green',clip_on=False)
plt.plot(dj,Dltd[:,1,1],c='purple',clip_on=False)

s = 17
w = 4
plt.scatter(dj[::w],Dltsp[:,0,2][::w],c='green',s=s,marker='o',clip_on=False)
plt.scatter(dj[::w],Dltd[:,1,1][::w],c='purple',s=s,marker='o',clip_on=False)

# plt.scatter(0.5,0.01785,c='b',zorder=2)

plt.plot(dj,Dltsp2[:,0,2],c='green')
plt.plot(dj,Dltd2[:,1,1],c='purple')

plt.scatter(dj[::w],Dltsp2[:,0,2][::w],c='green',s=s,marker='s',clip_on=False)
plt.scatter(dj[::w],Dltd2[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False)


plt.rc('text', usetex=True)


s = 4
leg = ['$g_t^{z}|\\Delta_{\\bot z}^{s\pm}|\quad p=(0,0)$','$g_t^x|\\Delta_{//x}^{d}|\\quad p=(0,0)$', \
       '$g_t^{z}|\\Delta_{\\bot z}^{s\pm}|\quad p=(0,-0.28)$','$g_t^x|\\Delta_{//x}^{d}|\\quad p=(0,-0.28)$']
plt.plot(0,Dltsp[0,0,2],'-o',label=leg[0],markersize=s,c='green',zorder=0)
plt.plot(0,Dltd[0,1,1],'-o',label=leg[1],markersize=s,c='purple',zorder=0)
plt.plot(0,Dltsp2[0,0,2],'-s',label=leg[2],markersize=s,c='green',zorder=0)
plt.plot(0,Dltd2[0,1,1],'-s',label=leg[3],markersize=s,c='purple',zorder=0)

plt.legend(frameon=False,fontsize=12,loc=(0.0,0.62))



plt.xlim(0,)
plt.ylim(0,)
plt.ylabel('$g_t|\\Delta^{\\alpha}|$',fontsize=17)
# plt.ylabel('$g_t^z|\\Delta^{s\pm}_{\\bot z}|$',fontsize=17)

plt.text(0.75,-0.015,'$J_{//}/J_{\\bot}$',fontsize=17)
plt.plot([0.5,0.5],[0,0.05],'--',c='k',lw=1,alpha=0.6)

plt.ylim(0,0.1)
plt.xlim(0,2)

plt.gca().set_aspect(18)


from matplotlib.ticker import AutoMinorLocator
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))

plt.savefig('J.pdf',bbox_inches = 'tight')


#%%
import numpy as np
import matplotlib.pyplot as plt
# from RMFT_FUNC import *
plt.rc('font', family='serif')
plt.rc('text', usetex=False)

def Gt(n,d):
    # assert np.all(n<=1 and n>=0), print(n); 'ERROR in n'
    # assert np.all((n>=0)*(n<=1)), 'ERROR in n'
    if np.any(n>1): 
        print('Warning: n '+str(n))
        n = 1
    # if n==0: return 1
    return (n>0)*1.*np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
    # assert np.all(n<=1 and n>=0), 'ERROR in n'
    assert np.all(n>=0), 'ERROR in n'
    # if n==0: return 1
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.

# -----------------------------------------------------
fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH0.npz'


dt = np.load(fn)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z

amp = np.abs

Dltd = amp(Dltd_inp )
Dlts = amp(Dlts_inp )
Dltsp = amp(Dlts_apc )


dj = Jr[:,0]/Jr[:,1]

#% -----------------------------------------------------
# fn2 = 'pams_figJ_p-0.15_0_complex_Nm220_0_2_96_JH0.npz'
fn2 = 'pams_figJ_p0_-0.1_complex_Nm220_0_2_96_JH0.npz'
fn2 = 'pams_figJ_p0_-0.28_complex_Nm220_0_2_96_JH0.npz'

dt = np.load(fn2)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z


Dltd2 = amp(Dltd_inp )
Dlts2 = amp(Dlts_inp )
Dltsp2 = amp(Dlts_apc )


# %---------------------------------------------------------
plt.plot(dj,Dltsp[:,0,2],c='green',clip_on=False)
plt.plot(dj,Dltd[:,1,1],c='purple',clip_on=False)

s = 17
w = 4
plt.scatter(dj[::w],Dltsp[:,0,2][::w],c='green',s=s,marker='o',clip_on=False)
plt.scatter(dj[::w],Dltd[:,1,1][::w],c='purple',s=s,marker='o',clip_on=False)

# plt.scatter(0.5,0.01785,c='b',zorder=2)

plt.plot(dj,Dltsp2[:,0,2],c='green')
plt.plot(dj,Dltd2[:,1,1],c='purple')

plt.scatter(dj[::w],Dltsp2[:,0,2][::w],c='green',s=s,marker='s',clip_on=False)
plt.scatter(dj[::w],Dltd2[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False)


plt.rc('text', usetex=True)


s = 4
leg = ['$g_t^{z}|\\Delta_{\\bot z}^{s\pm}|\quad p=(0,0)$','$g_t^x|\\Delta_{//x}^{d}|\\quad p=(0,0)$', \
       '$g_t^{z}|\\Delta_{\\bot z}^{s\pm}|\quad p=(0,-0.28)$','$g_t^x|\\Delta_{//x}^{d}|\\quad p=(0,-0.28)$']
plt.plot(0,Dltsp[0,0,2],'-o',label=leg[0],markersize=s,c='green',zorder=0)
plt.plot(0,Dltd[0,1,1],'-o',label=leg[1],markersize=s,c='purple',zorder=0)
plt.plot(0,Dltsp2[0,0,2],'-s',label=leg[2],markersize=s,c='green',zorder=0)
plt.plot(0,Dltd2[0,1,1],'-s',label=leg[3],markersize=s,c='purple',zorder=0)


plt.xlim(0,)
plt.ylim(0,)
plt.ylabel('$g_t|\\Delta^{\\alpha}|$',fontsize=17)
# plt.ylabel('$g_t^z|\\Delta^{s\pm}_{\\bot z}|$',fontsize=17)

plt.text(0.75,-0.015,'$J_{//}/J_{\\bot}$',fontsize=17)
plt.plot([0.5,0.5],[0,0.05],'--',c='k',lw=1,alpha=0.6)

plt.ylim(0,0.1)
plt.xlim(0,2)

plt.gca().set_aspect(18)


from matplotlib.ticker import AutoMinorLocator
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))


fn3 = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH-0.5.npz'
dt = np.load(fn3)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z


Dltd3 = amp(Dltd_inp )
Dlts3 = amp(Dlts_inp )
Dltsp3 = amp(Dlts_apc )

# plt.plot(dj,Dltsp3[:,0,2],'--',c='green',zorder=2,alpha=0.7)
plt.plot(dj,Dltd3[:,1,1],'--',c='purple',zorder=2,alpha=0.7)
# plt.scatter(dj[::w],Dltsp3[:,0,2][::w],c='green',s=s,marker='s',clip_on=False,zorder=2)
# plt.scatter(dj[::w],Dltd3[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False,zorder=2)
plt.legend(frameon=False,fontsize=12,loc=(0.0,0.62),ncol=1)

plt.text(1.27,0.006,'$J_H=-0.5$',fontsize=13)


plt.savefig('J.pdf',bbox_inches = 'tight')


#%%
import numpy as np
import matplotlib.pyplot as plt
# from RMFT_FUNC import *
plt.rc('font', family='serif')
plt.rc('text', usetex=False)

def Gt(n,d):
    # assert np.all(n<=1 and n>=0), print(n); 'ERROR in n'
    # assert np.all((n>=0)*(n<=1)), 'ERROR in n'
    if np.any(n>1): 
        print('Warning: n '+str(n))
        n = 1
    # if n==0: return 1
    return (n>0)*1.*np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
    # assert np.all(n<=1 and n>=0), 'ERROR in n'
    assert np.all(n>=0), 'ERROR in n'
    # if n==0: return 1
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.

# -----------------------------------------------------
fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH0.npz'
# fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH-0.5.npz'


dt = np.load(fn)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z

amp = np.abs

Dltd = amp(Dltd_inp )
Dlts = amp(Dlts_inp )
Dltsp = amp(Dlts_apc )


dj = Jr[:,0]/Jr[:,1]

#% -----------------------------------------------------
fn2 = 'pams_figJ_p0_0_complex_Nm220_0_2_96_J3-0.03_JH0.npz'


dt = np.load(fn2)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z


Dltd2 = amp(Dltd_inp )
Dlts2 = amp(Dlts_inp )
Dltsp2 = amp(Dlts_apc )


# %---------------------------------------------------------
plt.plot(dj,Dltsp[:,0,2],c='green',clip_on=False,label='$g_t^{\\bot z}|\\Delta^{\\bot z}_{s^\pm}|,\ J_{xz}=0$')
plt.plot(dj,Dltd[:,1,1],c='purple',clip_on=True,label='$g_t^{||x}|\\Delta^{||x}_{d}|,\ J_{xz}=0$')
plt.plot(dj,Dlts[:,1,1],':',c='g',clip_on=True,label='$g_t^{||x}|\\Delta^{||x}_{s^\pm}|,\ J_{xz}=0$')
plt.plot(dj,Dltd[:,0,0],':',c='purple',clip_on=True,label='$g_t^{||z}|\\Delta^{||z}_{d}|,\ J_{xz}=0$')

# plt.plot(dj,Dlts[:,0,0],'--',c='g',clip_on=True,label='$g_t^z|\\Delta_{//z}^{s\pm}|$')

# plt.plot(dj,Dltd[:,0,0],c='purple',clip_on=True,label='$g_t^x|\\Delta_{//z}^{d}|\\quad J_{xz}=0$')

s = 12
w = 4
# plt.scatter(dj[::w],Dltsp[:,0,2][::w],c='green',s=s,marker='o',clip_on=False)
# plt.scatter(dj[::w],Dltd[:,1,1][::w],c='purple',s=s,marker='o',clip_on=False)

# plt.scatter(0.5,0.01785,c='b',zorder=2)

# plt.plot(dj,Dlts2[:,1,1],':',c='g',clip_on=True,label='$g_t^x|\\Delta_{//x}^{s\pm}|\\quad J_{xz}=0.03$')
# plt.plot(dj,Dltsp2[:,0,2],'--',c='g',label='$g_t^{z}|\\Delta_{\\bot z}^{s\pm}|\\quad J_{xz}=0.03$')
# plt.plot(dj,Dltd2[:,1,1],'--',c='purple',label='$g_t^x|\\Delta_{//x}^{d}|\quad J_{xz}=0.03$')
# plt.plot(dj,Dltd2[:,0,0],'.-',c='purple',clip_on=True,label='$g_t^x|\\Delta_{//z}^{d}|\\quad J_{xz}=0.03$')

# plt.scatter(dj[::w],Dltsp2[:,0,2][::w],c='green',s=s,marker='s',clip_on=False)
# plt.scatter(dj[::w],Dltd2[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False)


plt.rc('text', usetex=True)




plt.xlim(0,)
plt.ylim(0,0.07)
plt.ylabel('$g_t|\\Delta^{\\alpha}|$',fontsize=17)
# plt.ylabel('$g_t^z|\\Delta^{s\pm}_{\\bot z}|$',fontsize=17)

plt.text(0.75,-0.015,'$J_{//}/J_{\\bot}$',fontsize=17)
plt.plot([0.5,0.5],[0,0.026],'--',c='grey',lw=2,alpha=0.6)


plt.xlim(0,2)

plt.gca().set_aspect(25)


from matplotlib.ticker import AutoMinorLocator
plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.gca().xaxis.set_minor_locator(AutoMinorLocator(2))




fn3 = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH-1.npz'
dt = np.load(fn3)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z


Dltd3 = amp(Dltd_inp )
Dlts3 = amp(Dlts_inp )
Dltsp3 = amp(Dlts_apc )

# plt.plot(dj,Dltsp3[:,0,2],'--',c='green',zorder=2,alpha=0.7)
plt.plot(dj,Dltd3[:,1,1],'-.',c='purple',zorder=2,alpha=0.7,label='$g_t^{||x}|\\Delta^{||x}_{d}|,\ J_{H}=-1$')
# plt.scatter(dj[::w],Dltsp3[:,0,2][::w],c='green',s=s,marker='s',clip_on=False,zorder=2)
# plt.scatter(dj[::w],Dltd3[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False,zorder=2)
plt.legend(frameon=False,fontsize=12,loc=(0.0,0.62),ncol=1)


# plt.legend(frameon=False,fontsize=13,loc=(0.0,0.66),ncol=2,columnspacing=0.6)


plt.savefig('J.pdf',bbox_inches = 'tight')

#%%
# fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH0.npz'
fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_J50.054_JH0.npz'
# fn = 'pams_figJ_p0_0_complex_Nm220_0_2_96_JH-0.5.npz'


dt = np.load(fn)
Jr = dt['Jr']
Pms = dt['Pms']

dlt0x,dlt0y,dlt0z = Pms[:,3],Pms[:,4],Pms[:,5]


N = dt['N']
Ns,d = 2,0
gt = np.einsum('ni,nj->nij',Gt(N*Ns,d),Gt(N*Ns,d))
#gJ = np.einsum('ni,nj->nij',GJ(N*Ns,d),GJ(N*Ns,d))
dlt0x *= gt
dlt0y *= gt
dlt0z *= gt
    
Dltd_inp = dlt0x-dlt0y
Dlts_inp = dlt0x+dlt0y
Dlts_apc = dlt0z

amp = np.abs

Dltd = amp(Dltd_inp )
Dlts = amp(Dlts_inp )
Dltsp = amp(Dlts_apc )


dj = Jr[:,0]/Jr[:,1]

plt.plot(dj,Dlts[:,1,1],':',c='g',clip_on=True,label='$g_t^x|\\Delta_{//x}^{s\pm}|$')
plt.plot(dj,Dlts[:,0,0],'--',c='g',clip_on=True,label='$g_t^z|\\Delta_{//z}^{s\pm}|$')
plt.plot(dj,Dltsp[:,0,2],c='green',clip_on=False,label='$g_t^{z}|\\Delta_{\\bot z}^{s\pm}|$')
plt.plot(dj,Dltd[:,1,1],':',c='purple',clip_on=True,label='$g_t^x|\\Delta_{//x}^{d}|$')
plt.plot(dj,Dltd[:,0,0],'--',c='purple',clip_on=True,label='$g_t^z|\\Delta_{//z}^{d}|$')

# plt.plot(dj,Dltd[:,0,0],c='purple',clip_on=True,label='$g_t^x|\\Delta_{//z}^{d}|\\quad J_{xz}=0$')


plt.legend(frameon=False,fontsize=12,loc=(0.05,0.42),ncol=1,columnspacing=0.6)
plt.ylim(0,0.08)
plt.xlim(0,2)
plt.gca().set_aspect(22)