from ngsolve import *
import numpy as np
from dispersion import *
import matplotlib.pyplot as plt
from netgen.occ import *


th_res = 1e-2 # 1e-3
maxh = 0.2 # 0.1
order = 2 # 3
nparam = 500
nval = 100


air = WorkPlane().RectangleC(1,1).Face()
rod = WorkPlane().Circle(0,0,0.11).Face()
mesh = CreateGeometry(air=air, rod=rod, max_h=maxh)
param_omega = Parameter(1)

mu, kappa = Permeability(freq=4.28, h0=0.16)
eps = lambda o:  15 + 8*sin(-np.pi/2 + o*2*np.pi)
min_omega = 1e-4
# max_omega = 0.625
max_omega = 0.71

omegas = np.linspace(min_omega, max_omega, nparam)

# plt.figure()
# plt.plot(omegas, [eps(o) for o in omegas]) #, label=r'$\varepsilon$')
# plt.plot(omegas, [15 for o in omegas], 'C1-')
# plt.xlabel('ωa/(2πc)')
# plt.ylabel(r'permittivity')
# # plt.legend()
# plt.tight_layout()
# # plt.show()
# # quit()


# omegas = np.concatenate([np.linspace(min_omega, 0.32, 350), np.linspace(0.4, 0.55, 850), np.linspace(0.55, max_omega, 100)])
dispQEP = DispersionQEP(mesh=mesh, mu=mu, kappa=kappa, eps=eps(param_omega), order = order, param_omega=param_omega)
print("QEP ndof = {}".format(dispQEP.fes.ndof))
dispQEP.th_imag = 1e-12
val, _, _ = dispQEP.CalculateBandStructure(params = omegas, nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega, buildRB=True, prefix='IV_QEP_', plot = True)
SortIntoBands(val, k_ind=[0,1,0], band_structure=True, plot=True)
# plt.show()
# quit()
# dispQEP.CalculateBandStructure(param=omegas,  nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega, buildRB=False)
omegas = np.linspace(min_omega, max_omega, nparam)
chern = dispQEP.ChernNumber_WLA(nzones=5, params=omegas, plot = True, buildRB=False, prefix='CN_IV_')
print(chern)
plt.show()