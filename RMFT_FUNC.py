import numpy as np
import matplotlib.pyplot as plt
from func import *
import itertools
# from H1band_hf import *
from H4band import *
from scipy.optimize import fsolve

# n0s = np.sum(n0)  # 1.5*2 two site, each site is 1.5

Fk1 = lambda x,y: 2*(np.cos(x)+np.cos(y))
Fk2 = lambda x,y: 4*np.cos(x)*np.cos(y)

Fkx = lambda x: 2*np.cos(x)
Fky = Fkx

Fkd = lambda x,y: 2*(np.cos(x)-np.cos(y))

Ns = 2


# def fsolve_tb(func,x0,args):
#     x00 = np.zeros_like(x0)
#     Nsize = len(x00)
#     y00 = func(x00,*args)
    
    
#     x_step = 0.1*np.ones([Nsize])
#     x = x00+x_step
#     y = func(x,*args)
#     dy = y-y00
    

# J1: inplane dx2-dx2 J2: apical dz2-dz2
# J3: inplane dx2-dz2 J4: inplane dz2-dz2 J4=0
# J5: apical dx2-dz2  J6: apical dx2-dx2  J5=J6=0
# chi0_00: <cdag_z2s c_z2s>  chi0_01: <cdag_z2s c_x2s>
# chi0_10: <cdag_x2s c_z2s>  chi0_11: <cdag_x2s c_x2s>
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

def Gt(n,d):
    # assert np.all(n<=1 and n>=0), print(n); 'ERROR in n'
    # assert np.all((n>=0)*(n<=1)), 'ERROR in n'
    # if np.any(n>1): 
    #     print('Warning: n '+str(n))
    #     n = 1
    if n==0: return 1
    return (n>0)*1.*np.sqrt((n-2*d)/np.abs(n-n**2/2))*(np.sqrt(np.abs(1-n+d))+np.sqrt(d)) + (n==0)*1.
def GJ(n,d):
    # assert np.all(n<=1 and n>=0), 'ERROR in n'
    assert np.all(n>=0), 'ERROR in n'
    if n==0: return 1
    return (n>0)*1.*(n-2*d)/(n-n**2/2) + (n==0)*1.

# def Gt(n,d):
#     # assert np.all(n<=1 and n>=0), print(n); 'ERROR in n'
#     assert np.all(n>=0), print(n); 'ERROR in n'
#     if n>1: n = 1
#     if n>0: return np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d))
#     return 1
# def GJ(n,d):
#     # assert np.all(n<=1 and n>=0), 'ERROR in n'
#     assert np.all(n>=0), print(n); 'ERROR in n'
#     if n>0: return (n-2*d)/(n-n**2/2)
#     return 1
# Gt = lambda n,d: np.sqrt((n-2*d)/(n-n**2/2))*(np.sqrt(1-n+d)+np.sqrt(d))
# GJ = lambda n,d: (n-2*d)/(1-n/2)/n
# Gt = lambda n,d: np.sqrt(2*(1-n)/(2-n))
# GJ = lambda n,d: 2/(2-n)

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

def get_params(hk,km,T=0,fc=float):
    x,y = km; Nm2 = km.shape[1]
    
    va,ve = np.linalg.eigh(hk)
    fv = ff(va,T=T)
    fkx,fky = Fkx(x),Fky(y)
    
    chi0x, chi0y, chi0z = np.zeros([Nb,Nb],fc), np.zeros([Nb,Nb],fc), np.zeros([Nb,Nb],fc)
    dlt0x, dlt0y, dlt0z = np.zeros([Nb,Nb],fc), np.zeros([Nb,Nb],fc), np.zeros([Nb,Nb],fc)
    for ib,jb in itertools.product(range(Nb),range(Nb)):
        if fc is float:
            chi0x[ib,jb] = np.sum((ve[:,ib]*ve[:,jb]*fv).sum(axis=-1)*fkx)/Nm2
            chi0y[ib,jb] = np.sum((ve[:,ib]*ve[:,jb]*fv).sum(axis=-1)*fky)/Nm2
            chi0z[ib,jb] = np.sum((ve[:,ib]*ve[:,jb]*fv).sum(axis=-1))/Nm2
            
            dlt0x[ib,jb] = np.sum((ve[:,ib]*ve[:,jb+Nb]*fv).sum(axis=-1)*fkx)/Nm2
            dlt0y[ib,jb] = np.sum((ve[:,ib]*ve[:,jb+Nb]*fv).sum(axis=-1)*fky)/Nm2
            dlt0z[ib,jb] = np.sum((ve[:,ib]*ve[:,jb+Nb]*fv).sum(axis=-1))/Nm2
        else:
            chi0x[ib,jb] = np.sum((ve[:,ib].conj()*ve[:,jb]*fv).sum(axis=-1)*fkx)/Nm2
            chi0y[ib,jb] = np.sum((ve[:,ib].conj()*ve[:,jb]*fv).sum(axis=-1)*fky)/Nm2
            chi0z[ib,jb] = np.sum((ve[:,ib].conj()*ve[:,jb]*fv).sum(axis=-1))/Nm2
            
            dlt0x[ib,jb] = np.sum((ve[:,ib].conj()*ve[:,jb+Nb]*fv).sum(axis=-1)*fkx)/Nm2
            dlt0y[ib,jb] = np.sum((ve[:,ib].conj()*ve[:,jb+Nb]*fv).sum(axis=-1)*fky)/Nm2
            dlt0z[ib,jb] = np.sum((ve[:,ib].conj()*ve[:,jb+Nb]*fv).sum(axis=-1))/Nm2

    return chi0x,chi0y,chi0z,dlt0x,dlt0y,dlt0z

def get_params_k(hk,km,T=0,fc=float):
    x,y = km; Nm2 = km.shape[1]
    
    va,ve = np.linalg.eigh(hk)
    fv = ff(va,T=T)
    fkx,fky = Fkx(x),Fky(y)
    
    chi0x, chi0y, chi0z = np.zeros((Nb,Nb)+x.shape,fc), np.zeros((Nb,Nb)+x.shape,fc), np.zeros((Nb,Nb)+x.shape,fc)
    dlt0x, dlt0y, dlt0z = np.zeros((Nb,Nb)+x.shape,fc), np.zeros((Nb,Nb)+x.shape,fc), np.zeros((Nb,Nb)+x.shape,fc),
    for ib,jb in itertools.product(range(Nb),range(Nb)):
        if fc is float:
            chi0x[ib,jb] = (ve[...,ib,:]*ve[...,jb,:]*fv).sum(axis=-1)*fkx
            chi0y[ib,jb] = (ve[...,ib,:]*ve[...,jb,:]*fv).sum(axis=-1)*fky
            chi0z[ib,jb] = (ve[...,ib,:]*ve[...,jb,:]*fv).sum(axis=-1)
            
            dlt0x[ib,jb] = (ve[...,ib,:]*ve[...,jb+Nb,:]*fv).sum(axis=-1)*fkx
            dlt0y[ib,jb] = (ve[...,ib,:]*ve[...,jb+Nb,:]*fv).sum(axis=-1)*fky
            dlt0z[ib,jb] = (ve[...,ib,:]*ve[...,jb+Nb,:]*fv).sum(axis=-1)
        else:
            chi0x[ib,jb] = (ve[...,ib,:].conj()*ve[...,jb,:]*fv).sum(axis=-1)*fkx
            chi0y[ib,jb] = (ve[...,ib,:].conj()*ve[...,jb,:]*fv).sum(axis=-1)*fky
            chi0z[ib,jb] = (ve[...,ib,:].conj()*ve[...,jb,:]*fv).sum(axis=-1)
            
            dlt0x[ib,jb] = (ve[...,ib,:].conj()*ve[...,jb+Nb,:]*fv).sum(axis=-1)*fkx
            dlt0y[ib,jb] = (ve[...,ib,:].conj()*ve[...,jb+Nb,:]*fv).sum(axis=-1)*fky
            dlt0z[ib,jb] = (ve[...,ib,:].conj()*ve[...,jb+Nb,:]*fv).sum(axis=-1)

    return chi0x,chi0y,chi0z,dlt0x,dlt0y,dlt0z
# assume A=B sublattice symmetry !!!!!!!!!!!!!!!!
def df1(mu,hk,ni,T): # mu is a float
    # print('mu=',mu)
    # Warning: fsolve has turn mu to a list, so I have to compitable with it
    mup = [mu[0]]*Nb+[-mu[0]]*Nb
    # print(mup)
    mut = np.diag(mup)
    dn = np.sum(density(hk-mut,T)[:2])-ni
    return dn
def df2(mu,hk,ni,T): 
    mup = np.array(mu)
    # print(mu)
    mut = np.diag(np.hstack((mup,mup,-mup,-mup)))
    # va = np.linalg.eigvalsh(hk-mut)
    dn = density(hk-mut,T)[:2]-ni[:2]
    return dn
# =============================================
def checkout(dn,eps=1e-3):
    if np.any(np.abs(dn)>eps):
        st = 'Wanring: dn= '
        for i in dn: st += '%.4f '%i
        print(st)

def loop(km,p,pms,Js,JH,n,T=0,fc=float):
    hk = Hmf(km,n,pms,Js,JH,fc=fc)
    
    # fix density =============================
    # If only one p is given, we will assume it to be the total doping for ONE Ni atom
    # If p is a list-like object, len(p)==Nb is required
    if isinstance(p,(float,int)):
        ni = np.sum(n0[:2])-np.array(p)/Ns
        muf = fsolve(df1,x0=1e-1,args=(hk,ni,T))[0]
        mu = np.array([muf]*Nb+[-muf]*Nb)
        # checkout(df1(muf,hk,ni,T))
    elif isinstance(p, (list,tuple,np.ndarray)) and len(p)==Nb:
        ni = np.array(n0)-np.array(p)/Ns
        muf = fsolve(df2,x0=[1e-1]*2,args=(hk,ni,T))
        mu = np.hstack((muf,muf,-muf,-muf)) 
        # checkout(df2(muf,hk,ni,T))
        
    assert not np.any(np.isnan(mu)), 'ERROR in mu'
    hk -= np.diag(mu)
    
    n = density(hk,T=T)[:Nb]
    # dn = (n0-n)*Ns
    
    pms = get_params(hk,km,T=T,fc=fc)
    return pms,mu,n

def scf(km,p,pms0,Js,JH=0,eps=1e-3,T=0,fc=float,n=n0,calcF=False):
    err = 999
    lp = 0; Fi = []; Mu = []
    
    while err>eps:
        pms = pms0
        pms,mu,n = loop(km,p,pms,Js,JH,n,T,fc=fc)

        # print('nnnn=',n)
        de = np.abs(pms)-np.abs(pms0)
        err = np.average(np.abs(de))
        # print(' %2d err= %4f'%(lp,err)+' pms=[%.6f %.6f %.6f %.6f]'%pms+' n=[%.2f %.2f]'%tuple(n))
        print(' %2d err= %4f'%(lp,err), 'n=',n)
        # print('n=',n)
        pms0 = pms
        # mu00 = mu[:Nb]
        lp += 1
        
        if calcF:
            # print('n3=',n)
            Fi.append( F(mu,T,pms,Js,n,km) )
            Mu.append(mu)
            print('Free energy: %g'%np.sum(Fi[-1]))
            
        if False: # for testing !!!!!!!!!!!!!
            dz,dx = 1e-6,0
            muu = mu+[dz,dx,dz,dx,-dz,-dx,-dz,-dx]
            mud = mu-[dz,dx,dz,dx,-dz,-dx,-dz,-dx]
            Fu = F(muu,T,pms,Js,n,km)
            Fd = F(mud,T,pms,Js,n,km)
            nzt = -np.sum(Fu-Fd)/dz/2 /2  # WARNING: 2 due to repeat of 2 orbitals 
            
            dz,dx = 1e-6,1e-6
            muu = mu+[dz,dx,dz,dx,-dz,-dx,-dz,-dx]
            mud = mu-[dz,dx,dz,dx,-dz,-dx,-dz,-dx]
            Fu = F(muu,T,pms,Js,n,km)
            Fd = F(mud,T,pms,Js,n,km)
            nxt = -np.sum(Fu-Fd)/dx/2 /2
            
            print('  nzt, nxt= %g  %g'%(nzt,nxt))
            
            dp = 1e-6
            NF = len(Fu)
            dFp = np.zeros([6,NF+1])  #last one store the sum of dF/dp
            ij = [[1,1,3,3],[1,1,3,3],[0,2,2,0],\
                  [1,1,3,3],[1,1,3,3],[0,2,2,0]]
            for i in range(6):
                i1,j1,i2,j2 = ij[i]
                pmsu = np.copy(pms); pmsd = np.copy(pms)
                pmsu[i][i1,j1] += dp; pmsu[i][i2,j2] += dp
                pmsd[i][i1,j1] -= dp; pmsd[i][i2,j2] -= dp
                Fu = F(mu,T,pmsu,Js,n,km)
                Fd = F(mu,T,pmsd,Js,n,km)
                dFp[i,:-1] = (Fu-Fd)/2/dp
                dFp[i,-1] = np.sum(dFp[i,:-1])
            print('  dFp/dp =',dFp)
            
        if np.max(np.abs(np.array(pms)))<1e-20: break
    if not calcF: return pms,mu,n
    else: return pms,mu,n,np.array(Fi),np.array(Mu)
    
def getHk(km,n,pms,Js,mu,JH=0):
    if isinstance(pms[0].flat[0], complex): fc = complex
    else: fc = float
    hk = Hmf(km,n,pms,Js,JH,fc=fc)-np.diag(mu)
    return hk

def logcosh(x):
    # s always has real part >= 0
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)
def log1pexp(x):
    return logcosh(x/2)+x/2+np.log(2)
    
# mu=[muz,mux]
def F(mu,T,pms,Js,n,km):
    
    Nm2 = km.shape[1]
    
    # print('n0=',n)
    hkm = getHk(km,n,pms,Js,mu)
    # print('n1=',n)
    hkm0 = hkm.diagonal(axis1=-2,axis2=-1)[:,:4]
    vam, vem = np.linalg.eigh(hkm)

    # ppm = np.exp(-vam/T)+1
    
    if T==0: T=1e-10
    # F10 = -T*np.log(2+2*np.cosh(vam[:,:4]/T)).sum()/Nm2
    F1 = -T*log1pexp(-vam/T).sum()/Nm2
    # print(F10,F1)
    
    F2 = np.real(hkm0).sum()/Nm2
    
    Ns,d = 2,0
    gz,gx = GJ(n[0]*Ns,d)**2, GJ(n[1]*Ns,d)**2
    
    pabs = np.abs(pms)
    F31 = Js[0]*3/2*(pabs[0,1,1]**2+pabs[1,1,1]**2)*2 *gx/4
    F32 = Js[1]*3/2*pabs[2,0,2]**2 *gz/4 *2
    F41 = Js[0]*3/2*(pabs[3,1,1]**2+pabs[4,1,1]**2)*2 *gx/4
    F42 = Js[1]*3/2*pabs[5,0,2]**2 *gz/4 *2
    
    # NEW !!!!
    
    
    
    # F = F1+F2+F3+F4

    return np.array([F1,F2,F31,F32,F41,F42])


if __name__=='__main__':
    fc = complex
    
    if fc is float: ph = 1.
    else: ph = np.exp(1j*0.1)
    
    chi0x,chi0y,chi0z = np.ones([Nb,Nb],fc)*0.1, np.ones([Nb,Nb],fc)*0.11, np.ones([Nb,Nb],fc)*0.14
    # dlt0x,dlt0y,dlt0z = np.ones([Nb,Nb],fc)*0.1, -np.ones([Nb,Nb],fc)*0.11*ph, np.ones([Nb,Nb],fc)*0.14*ph
    dlt0x,dlt0y,dlt0z = np.ones([Nb,Nb],fc)*0., -np.ones([Nb,Nb],fc)*0.*ph, np.ones([Nb,Nb],fc)*0.*ph

    pms0 = chi0x,chi0y,chi0z,dlt0x,dlt0y,dlt0z
    
    # pms0 = [np.zeros([Nb,Nb],fc) for _ in range(6)]
    
    # p = 0., 0., 0., 0.
    p = 0
    # Js = 0.2,0.2,0,0,0,0
    Js = 0,0,0,0,0,0.2
    JH = 0.
    # nt = 0.2,0.4,0.2,0.4
    Nm = 220; Nm2 = Nm**2
    km = kmesh([Nm,Nm],flatten=True)
    # hk = Hmf(km,n0,pms0,Js,JH,fc=fc)
    # print(density(hk,T=0))
    # va,ve = np.linalg.eigh(hk)
    # pms = get_params(hk,km,T=0,fc=fc)
    # print(pms[0])
    print(n0)
    
    #%%
    pz0,px0 = 1-n0[:2]*2
    rz,rx = pz0/(pz0+px0), px0/(pz0+px0)
    p00 = 0.*rz,0.*rx
    # p00 = -0.06,0.44
    
    # pp = np.array([0.417691,0.33139337])-np.array(p00)/2
    # pp0 = (n0[:2]-pp)*2
    # p0 = pp0[0], pp0[1]
    # p0 = 0.37*rz,0.37*rx
    nz,nx = 0.93, 1.5-0.93
    p0 = n0[0]*2-nz,n0[1]*2-nx
    # p = 0,0.1,0,0.1
    p = p0+p0
    # p = 0
    Js = 0.01,0.1,0.0,  0,0,0.
    # Js = 0.09,0.18,0.0,  0,0,0.
    JH = 0.0
    # pms0 = 0.1,0.3,0.2,-0.1
    # pms,mu,n = loop(km,p,Js,pms0)
    pms,mu,n = scf(km,p,pms0,Js,JH,eps=1e-5,fc=complex)
    
    fn = 'pms_p%g_%g_J%g_%g_%g_nosc.npz'%(p[0],p[1],Js[0],Js[1],Js[2])
    np.savez(fn,pms=np.array(pms),mu=mu,n=n,Js=Js,p=p)
    
    # # v = 1
    # fn = 'pms_p%g_%g_VxVxz%g.npz'%(p00[0],p00[1],v)
    # np.savez(fn,pms=np.array(pms),mu=mu,n=n,Js=Js,p=p)
    
    # print(density(hk))
    print(pms[0][1,1],pms[1][1,1])
    print(pms[3][1,1],pms[4][1,1])
    print(pms[5][0,2])    #%% 
    # pms00 = [np.zeros([Nb,Nb]) for _ in range(6)]
    # hk,mu = Hmf(km,n0,pms00,Js)
    # Fii = F(mu,T,pms,Js,n,km)
    #%%
    hk = getHk(km,n,pms,Js,mu)
    hki = hk[0]
    va0,ve0 = np.linalg.eigh(hki)
    va,ve = np.linalg.eigh(hki[:4,:4])
    
    vap = ve0.T.conj().dot(hk[200,:Nb,:Nb]).dot(ve0)
    
    # Dltdiag = ve.dot(hki[:4,4:]).dot(ve.T.conj())
    Dltdiag = ve.T.conj().dot(hki[:4,4:]).dot(ve)
    
    # hkd,hkod = hki[:4,:4],hki[:4,4:]
    # vap,vep = np.linalg.eigh(hkd.dot(hkd.T.conj())+hkod.dot(hkod.conj().T) )
    # hkp = np.diag(np.sqrt(vap)).dot(vep)
    # vaq,veq = np.linalg.eigh(hkp)
    #%%
    # fn = 'pms_p-0.15_0_J0.09_0.18_0.npz'
    # dt = np.load(fn)
    # pms,mu,n,Js,p = dt['pms'],dt['mu'],dt['n'],dt['Js'],dt['p']
    pi = np.pi
    fig,ax = plt.subplots(2,2,figsize=(4,4))
    # plt.subplot(221)
    plt.sca(ax[0,0])
    Nk0 = 100
    kp = kpath(['G','X','M','G'],Nk0,ndim=2); Nkp = kp.shape[1]
    hkp = getHk(kp,n,pms,Js,mu)
    Eplot(hkp,cweight=[0,1,2,3],cmap='Greys',s=2)
    # Eplot(hkp)
    # Eplot(hkp)
    # hkp0 = Ht(*kp)
    # Eplot(hkp0)
    # plt.ylim(-0.45,0.45)
    # plt.colorbar()
    plt.axhline(0,color='r',linestyle='--')
    plt.ylim(-1,1)
    plt.xticks([0,Nk0,Nk0*2,Nkp-1],['G','X','M','G'])
    plt.grid(axis='x')
    plt.gca().set_aspect(140)
    # plt.show()
    
    # plt.subplot(222)
    plt.sca(ax[0,1])
    Nmfs = 220
    kmfs = kmesh([Nmfs,Nmfs],xlim=(-pi,pi),ylim=(-pi,pi),close=True,flatten=False)
    hkmfs = getHk(kmfs,n,pms,Js,mu)
    fs2dplot(hkmfs,km=kmfs,brd=0.025,cweight=[1,3,5,7],vmin=0,vmax=1,cmap='jet')
    # fs2dplot(hkm,km=kmfs,brd=0.08,vmin=0,vmax=1,cmap='jet')
    
    # plt.colorbar(shrink=0.6)
    plt.plot([-pi,pi],[pi,-pi],c='r')
    plt.plot([pi,-pi],[pi,-pi],c='r')
    plt.plot([pi,0],[0,pi],c='b')
    plt.plot([pi,0],[0,-pi],c='b')
    plt.plot([-pi,0],[0,pi],c='b')
    plt.plot([-pi,0],[0,-pi],c='b')
    plt.xlim(-pi,pi)
    plt.ylim(-pi,pi)
    
    plt.suptitle('px=%g pz=%g J1=%g J2=%g J3=%g'%(p[1],p[0],Js[0],Js[1],Js[2]),y=0.94)
    # plt.suptitle('V=3.5 p=0.05',y=0.8)
    
    # for non-interacting FS
    # plt.subplot(223)
    hk = Hk(*km)
    mu2 = find_mu(hk,n=n,mu0=mu[:4])
    
    
    plt.sca(ax[1,0])
    hkp = Hk(*kp)
    Eplot(hkp-np.diag(mu2),cweight=[1,3],cmap='jet',s=2)
    plt.axhline(0,color='r',linestyle='--')
    plt.ylim(-1,2.5)
    plt.xticks([0,Nk0,Nk0*2,Nkp-1],['G','X','M','G'])
    plt.grid(axis='x')
    plt.gca().set_aspect(80)
    
    # plt.subplot(224)
    plt.sca(ax[1,1])
    hkmfs2 = Hk(*kmfs)-np.diag(mu2)
    
    fs2dplot(hkmfs2,km=kmfs,brd=0.025,cweight=[1,3],vmin=0,vmax=1,cmap='jet')
    # fs2dplot(hkm,km=kmfs,brd=0.08,vmin=0,vmax=1,cmap='jet')
    
    # plt.colorbar(shrink=0.6)
    plt.plot([-pi,pi],[pi,-pi],c='r')
    plt.plot([pi,-pi],[pi,-pi],c='r')
    plt.plot([pi,0],[0,pi],c='b')
    plt.plot([pi,0],[0,-pi],c='b')
    plt.plot([-pi,0],[0,pi],c='b')
    plt.plot([-pi,0],[0,-pi],c='b')
    plt.xlim(-pi,pi)
    plt.ylim(-pi,pi)
    
    #%%
    Nmfs = 120
    kmfs = kmesh([Nmfs,Nmfs],xlim=(-pi,pi),ylim=(-pi,pi),close=False,flatten=False)
    hkmfs = getHk(kmfs,n,pms,Js,mu)
    pmsk = get_params_k(hkmfs, km=kmfs,fc=fc)
    
    amp = np.real
    # amp = np.real
    Dltdk = amp(pmsk[3]-pmsk[4])
    Dltsk = amp(pmsk[3]+pmsk[4])
    Dltspk = amp(pmsk[5])
    
    Dltd = amp((pmsk[3]-pmsk[4]).sum(axis=(-1,-2))/Nmfs**2)
    Dlts = amp((pmsk[3]+pmsk[4]).sum(axis=(-1,-2))/Nmfs**2)
    Dltsp = amp(pmsk[5].sum(axis=(-1,-2))/Nmfs**2)
    
    vmin,vmax = -0.5, 0.5
    cmap = 'seismic'
    fig,ax = plt.subplots(2,3,sharex=True,sharey=True)
    ax[0,0].pcolormesh(*kmfs,Dltdk[0,0],edgecolor='face',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[0,0].set_xlabel('%.4f'%Dltd[0,0])
    ax[1,0].pcolormesh(*kmfs,Dltdk[1,1],edgecolor='face',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[1,0].set_xlabel('%.4f'%Dltd[1,1])
    
    ax[0,1].pcolormesh(*kmfs,Dltsk[0,0],edgecolor='face',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[0,1].set_xlabel('%.4f'%Dlts[0,0])
    ax[1,1].pcolormesh(*kmfs,Dltsk[1,1],edgecolor='face',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[1,1].set_xlabel('%.4f'%Dlts[1,1])
    
    ax[0,2].pcolormesh(*kmfs,Dltspk[0,2],edgecolor='face',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[0,2].set_xlabel('%.4f'%Dltsp[0,2])
    im = ax[1,2].pcolormesh(*kmfs,Dltspk[1,3],edgecolor='face',cmap=cmap,vmin=vmin,vmax=vmax)
    ax[1,2].set_xlabel('%.4f'%Dltsp[1,3])
    
    cax = fig.add_axes([0.92, 0.08, 0.04, 0.7])
    fig.colorbar(im, cax=cax,cmap=cmap)
    
    ax[0,0].set_ylabel('dz2-dz2')
    ax[1,0].set_ylabel('dx2-dx2')
    ax[0,0].text(-1,3.5,'d-wave')
    ax[0,1].text(-1,3.5,'s-wave')
    ax[0,2].text(-1,3.5,'apical s-wave')

    plt.suptitle('px=%g pz=%g J1=%g J2=%g J3=%g'%(p[1],p[0],Js[0],Js[1],Js[2]),y=1)
    # plt.gca().set_aspect(1)
    #%%
    Js = 0.2,0.,0,0,0,0.
    # phase = np.exp(1j*0.1)
    # phase = 1
    
    # pms = pms0
    Pms,Mu,N = [],[],[]
    
    Nps = 20
    ps = np.linspace(0.,0.35,Nps)
    for ip,p in enumerate(ps):
        print('ip= %d/%d ==============='%(ip,Nps))
        pi = 0,p,0,p
        # pi = p
        pms,mu,n = scf(km,pi,pms0,Js,eps=2e-5,fc=fc)
        Pms.append(pms); Mu.append(mu); N.append(n)
        
        pms0 = pms
        
    # np.savez()
    
    #%%
    Pms = np.array(Pms); Mu = np.array(Mu)
    plt.plot(ps,Pms[:,0,1,1],'-x',c='b')
    # plt.plot(ps,Pms[:,1,1,1],'-o',c='b')
    plt.plot(ps,Pms[:,3,1,1],'-x',c='r')
    # plt.plot(ps,Pms[:,4,1,1],'-o',c='r')

    gDlt = 2*ps/(1+p)*Pms[:,3,1,1]
    plt.plot(ps,gDlt)
    plt.ylim(0,0.4)
    plt.xlim(0)
    # plt.text(0.05,0.1,'gt=2p/(1+p),gJ=4/(1+p)^2')
    plt.gca().set_aspect(0.8)
    
    #%%
    dt = np.loadtxt('RMFT_oneband.txt')
    plt.plot(dt[:,0],dt[:,1:])
    #%%
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    
    Pms = np.array(Pms); Mu = np.array(Mu)
    dw = Pms[:,3,1,1]-Pms[:,4,1,1]
    sw = Pms[:,3,1,1]+Pms[:,4,1,1]
    spw = Pms[:,5,1,3]
    plt.plot(ps,dw,label='d-wave')
    plt.plot(ps,sw,label='s-wave')
    plt.plot(ps,spw,'-x',label='apical s-wave')
    plt.legend(frameon=False)
    plt.xlabel('p')
    # gDlt = 2*ps/(1+p)*Pms[:,3,1,1]
    # plt.plot(ps,gDlt)
    plt.ylim(0,0.5)
    plt.xlim(0)
    # plt.text(0.05,0.1,'gt=2p/(1+p),gJ=4/(1+p)^2')
    plt.text(0.05,0.1,'$V_\\bot=0.,\ J_\\bot=0.2$')
    plt.gca().set_aspect(0.7)
    
    