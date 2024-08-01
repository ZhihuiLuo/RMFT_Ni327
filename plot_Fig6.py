import numpy as np
import matplotlib.pyplot as plt
# from RMFT_FUNC import *
plt.rc('font', family='serif')
plt.rc('text', usetex=False)

def Gt(n,d):
    if np.any(n>1): 
        print('Warning: n '+str(n))
        n = 1
    # if n==0: return 1
    return (n>0)*1.*np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
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
plt.plot(dj,Dltsp[:,0,2],c='green',clip_on=False,label='$g_t^{\\bot z}|\\Delta^{\\bot z}_{s^\pm}|\quad J_{xz}=0$')
plt.plot(dj,Dltd[:,1,1],c='purple',clip_on=True,label='$g_t^{||x}|\\Delta^{||x}_{d}|\quad J_{xz}=0$')
plt.plot(dj,Dlts[:,1,1],':',c='g',clip_on=True,label='$g_t^{||x}|\\Delta^{||x}_{s^\pm}|\quad J_{xz}=0$')
plt.plot(dj,Dltd[:,0,0],':',c='purple',clip_on=True,label='$g_t^{||z}|\\Delta^{||z}_{d}|\quad J_{xz}=0$')

# plt.plot(dj,Dlts[:,0,0],'--',c='g',clip_on=True,label='$g_t^z|\\Delta_{//z}^{s\pm}|$')

# plt.plot(dj,Dltd[:,0,0],c='purple',clip_on=True,label='$g_t^x|\\Delta_{//z}^{d}|\\quad J_{xz}=0$')

s = 12
w = 4
# plt.scatter(dj[::w],Dltsp[:,0,2][::w],c='green',s=s,marker='o',clip_on=False)
# plt.scatter(dj[::w],Dltd[:,1,1][::w],c='purple',s=s,marker='o',clip_on=False)

# plt.scatter(0.5,0.01785,c='b',zorder=2)

# plt.plot(dj,Dlts2[:,1,1],'--',c='g',clip_on=True,label='$g_t^x|\\Delta_{//x}^{s\pm}|\\quad J_{xz}=0.03$')
plt.plot(dj,Dltsp2[:,0,2],'--',c='g',label='$g_t^{\\bot z}|\\Delta^{\\bot z}_{s^\pm}|\\quad J_{xz}=0.03$')
plt.plot(dj,Dltd2[:,1,1],'--',c='purple',label='$g_t^{||x}|\\Delta^{||x}_{d}|\quad J_{xz}=0.03$')
# plt.plot(dj,Dltd2[:,0,0],'.-',c='purple',clip_on=True,label='$g_t^x|\\Delta_{//z}^{d}|\\quad J_{xz}=0.03$')

# plt.scatter(dj[::w],Dltsp2[:,0,2][::w],c='green',s=s,marker='s',clip_on=False)
# plt.scatter(dj[::w],Dltd2[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False)


plt.rc('text', usetex=True)




plt.xlim(0,)
plt.ylim(0,0.07)
plt.ylabel('$g_t^\\nu|\\Delta^{\\nu}_\ell|$',fontsize=17)
# plt.ylabel('$g_t^z|\\Delta^{s\pm}_{\\bot z}|$',fontsize=17)

plt.text(0.75,-0.015,'$J_{||}/J_{\\bot}$',fontsize=17)
# plt.plot([0.5,0.5],[0,0.026],'--',c='grey',lw=2,alpha=0.6)
plt.arrow(0.5,0.02,0.0,-0.015,lw=2,color='grey',head_length=0.005,head_width=0.03)


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
plt.plot(dj,Dltd3[:,1,1],'-.',c='purple',zorder=2,alpha=0.7,label='$g_t^{||x}|\\Delta^{||x}_{d}|\quad J_{H}=-1$')
# plt.scatter(dj[::w],Dltsp3[:,0,2][::w],c='green',s=s,marker='s',clip_on=False,zorder=2)
# plt.scatter(dj[::w],Dltd3[:,1,1][::w],c='purple',s=s,marker='s',clip_on=False,zorder=2)
plt.legend(frameon=False,fontsize=12,loc=(0.0,0.32),ncol=1)


# plt.legend(frameon=False,fontsize=13,loc=(0.0,0.66),ncol=2,columnspacing=0.6)


plt.savefig('Fig6.pdf',bbox_inches = 'tight')