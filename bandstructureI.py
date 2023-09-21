from ngsolve import *
import numpy as np
from dispersion import *
import matplotlib.pyplot as plt
from netgen.occ import *


out = 'output/'
th_res = 1e-3
maxh = 0.1
order = 3
nval = 50 
nparam = 300 #3000

air = WorkPlane().RectangleC(1,1).Face()
rod = WorkPlane().Circle(0,0,0.11).Face()

mesh = CreateGeometry(air=air, rod=rod, max_h=maxh)

mu, kappa = Permeability(freq=4.28, h0=0.16)
param_omega=Parameter(1)
eps = 15
min_omega = 1e-4
max_omega = 0.71 #0.625 or 0.71
nbands = 4

dispGEP = DispersionGEP(mesh=mesh, mu=mu, kappa=kappa, eps=eps, order = order)
dispQEP = DispersionQEP(mesh=mesh, mu=mu, kappa=kappa, eps=eps, order = order, param_omega=param_omega)

print("GEP ndof = {}, QEP ndof = {}".format(dispGEP.fes.ndof, dispQEP.fes.ndof))

dispGEP.CalculateBandStructure(nbands=nbands, nparam=nparam, nval=nval, th_res = th_res, min_omega=min_omega, max_omega = max_omega, useRB=False, prefix = 'I_GEP_full_')
dispGEP.CalculateBandStructure(nbands=nbands, nparam=nparam, nval=nval, th_res = th_res, min_omega=min_omega, max_omega = max_omega, prefix = 'I_GEP_')
dispQEP.CalculateBandStructure(nparam=nparam,  nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega) 

plt.show()

omegas = np.concatenate([np.linspace(min_omega, 0.35, 2000), np.linspace(0.42, max_omega, 8000)])
dispQEP.CalculateBandStructure( params=omegas, nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega, buildRB=False,  prefix = 'I_QEP_')

plt.show()