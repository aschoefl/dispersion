from ngsolve import *
import numpy as np
from dispersion import *
import matplotlib.pyplot as plt
from netgen.occ import *

'''benchmark example, takes a long time to compute'''

th_res = 1e-2 # 1e-3
maxh = 0.1 #0.1
order = 3
nval = 100
nparam = 500 #10000

c = 299792458
param_omega = Parameter(1)
omega_p = 2*np.pi*1914*1e12*5e-6
omega_tau =  2*np.pi*8.34*1e12*5e-6
f = lambda o: c*2*np.pi*o*5e-6
air = WorkPlane().RectangleC(1,1).Face()
rod = WorkPlane().Circle(0,0,0.2).Face()
eps = lambda o: (1- (omega_p**2)/(f(o)**2-1j*f(o)*omega_tau))
mu, kappa = Permeability(freq=0, h0=0)
# print(mu, kappa)
mesh = CreateGeometry(air=air, rod=rod, max_h=maxh)

min_omega = 0.5
max_omega = 1.6
# omegas = np.concatenate([np.linspace(min_omega, 0.3, 2000), np.linspace(0.38, max_omega, 8000)])
# omegas = np.concatenate([np.linspace(min_omega, 0.3, 20), np.linspace(0.38, max_omega, 500)])
omegas = np.linspace(min_omega, max_omega, nparam)

plt.figure()
plt.plot(omegas, [eps(o) for o in omegas]) #, label=r'$\varepsilon$')
plt.xlabel('ωa/(2πc)')
plt.ylabel(r'permittivity')
# plt.legend()
plt.tight_layout()
# plt.show()

dispQEP = DispersionQEP(mesh=mesh, mu=mu, kappa=kappa, eps=eps(param_omega), order = order, param_omega=param_omega)
dispQEP.logging = False
dispQEP.th_imag = 1e-3
dispQEP.nr_eigs = 100

print("QEP ndof = {}".format(dispQEP.fes.ndof))
dispQEP.CalculateBandStructure(params=omegas, nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega, buildRB=True, prefix = 'II_QEP_')
# plt.show()
# dispQEP.CalculateBandStructure(params=omegas,  nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega, useRB=False, prefix = 'II_QEP_full')
# print(dispQEP.snapshot_f)

plt.show()