import numpy as np
pi = np.pi

def kmesh(kl,xlim=(0,2*pi),ylim=(0,2*pi),zlim=(0,2*pi),mv=0,close=False,flatten=False):
    dim = len(kl)
    if dim==1: Nx= kl[0]
    elif dim==2: Nx,Ny = kl
    elif dim==3: Nx,Ny,Nz = kl
    else: raise Exception('Too many args')
    
    if close: endx = Nx*1j
    else: endx = (xlim[1]-xlim[0])/Nx
    if dim==1:     
        kx = np.mgrid[xlim[0]:xlim[1]:endx]
        km = (kx+mv)[np.newaxis]
    elif dim==2:
        if close: endy = Ny*1j
        else: endy = (ylim[1]-ylim[0])/Ny
        kx,ky = np.mgrid[xlim[0]:xlim[1]:endx, \
                         ylim[0]:ylim[1]:endy]
        km = np.stack((kx+mv,ky+mv))
    elif dim==3:
        if close: endy = Ny*1j; endz = Nz*1j
        else: endy = (ylim[1]-ylim[0])/Ny; endz = (zlim[1]-zlim[0])/Nz
        kx,ky,kz = np.mgrid[xlim[0]:xlim[1]:endx, \
                            ylim[0]:ylim[1]:endy, \
                            zlim[0]:zlim[1]:endz]
        km = np.stack((kx+mv,ky+mv,kz+mv))
    if flatten: km = km.reshape([dim,np.int_(np.prod(kl))])
    return km