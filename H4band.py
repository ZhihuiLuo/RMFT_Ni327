import numpy as np
from func import *

Nb = 4
n0 = [0.41805728, 0.33223161, 0.41805728, 0.33223161]  # density(Htb)


# dz2 dx2 dz2 dx2
def Hk(kx,ky,fct=1):
    m = np.zeros(np.shape(kx)+(4,4))
    tx1 = -0.110
    tx2 = -0.483
    Vx = 0.239
    txy1 = -0.017
    txy2 = 0.069
    V1 = -0.635; V2 = 0.005
    Vxz = -0.034
    E1,E2 = 0.409,0.776
    E0 = [E1,E2,E1,E2]
    # ----------------------------------------------
    m[...,0,0] = 2*tx1*(np.cos(kx)+np.cos(ky)) +4*txy1*np.cos(kx)*np.cos(ky)
    m[...,1,1] = 2*tx2*(np.cos(kx)+np.cos(ky)) +4*txy2*np.cos(kx)*np.cos(ky)
    m[...,0,1] = 2*Vx*(np.cos(kx)-np.cos(ky)) *fct
    m[...,1,0] = m[...,0,1]
    
    m[...,0,3] = 2*Vxz*(np.cos(kx)-np.cos(ky)) *fct
    m[...,1,2] = m[...,0,3]
    
    m[...,0,2] = V1 *1
    m[...,1,3] = V2 *1
    
    m[...,2:,2:] = m[...,:2,:2]
    m[...,2:,:2] = m[...,:2,2:]
    
    m += np.diag(E0)
    return m



N = 200
km = kmesh([N,N],close=False,flatten=True)
hkm = Hk(*km)
n0 = density(hkm,T=0)
print('n0=',n0)
