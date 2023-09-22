
from ngsolve import *
import numpy as np
from dispersion import *
import matplotlib.pyplot as plt
from netgen.occ import *

th_res = 1e-3
maxh = 0.1
order = 3
nval = 100
nparam = 500

# DispersionQEP.logging = True

param_omega = Parameter(1)
air = WorkPlane().RectangleC(1,1).Face()
rod = WorkPlane().Circle(0,0,0.11).Face()
mesh = CreateGeometry(air=air, rod=rod, max_h=maxh)

min_omega = 0.2 #1e-4 #0
max_omega = 0.875 #75

eps = 15
ms = 1780*1e-4 
h0 = 0.16
gamma = 1.75784*1e11 #1.759*1e11 
o0  = gamma*h0
om  = gamma*ms
c = 299792458

fac = 200
f = lambda o: fac*2*np.pi*o*c #/0.01
kappa = lambda o: om*f(o)/(o0**2-f(o)**2) #+1.5*sin(4*np.pi*o)
mu = lambda o: (1+ om*o0/(o0**2-f(o)**2))
sing = o0/(2*np.pi*c*fac)
delta = 0.05
print('singularity at omega = {} '.format(o0/(2*np.pi*c*fac)))

plt.figure()

if sing > min_omega:
    omegas = np.linspace(min_omega, sing-delta, 150)
    omegas_test = omegas
    plt.plot(omegas, [kappa(o) for o in omegas], 'C0-')
    plt.plot(omegas, [mu(o) for o in omegas], 'C1-')
    omegas = np.linspace( sing+delta, max_omega, 200)
    omegas_test = np.concatenate([omegas_test, omegas])
else:
    omegas = np.linspace( 0.1, max_omega, 200)
    omegas_test = np.concatenate([np.linspace( min_omega, 0.38, 500), np.linspace( 0.49, max_omega, 1500)])



plt.plot(omegas, [kappa(o) for o in omegas], 'C0-',  label = r'$\kappa$')
plt.plot(omegas, [mu(o) for o in omegas], 'C1-', label=r'$\mu$')
plt.plot(omegas, [1 for o in omegas], 'C7--')
plt.plot(omegas, [0 for o in omegas], 'C7--')

plt.xlabel('ωa/(2πc)')
plt.legend()
# plt.show()
# quit()

mesh = CreateGeometry(air=air, rod=rod, max_h=maxh)
dispQEP = DispersionQEP(mesh=mesh, mu=mu(param_omega), kappa=kappa(param_omega), eps=eps, order = order, param_omega=param_omega)
# dispQEP.th_imag = 1e-14
dispQEP.th_imag = 1e-12 # better results with 1e-14 for building the RB and 1e-12 for calculating values
print("QEP ndof = {}".format(dispQEP.fes.ndof))
dispQEP.CalculateBandStructure(params=omegas_test, nval=nval, th_res=th_res, min_omega=min_omega, max_omega = max_omega, buildRB=True, useRB=True, plot = True, prefix='III_QEP_')
# dispQEP.CalculateBandStructure(nparam=1000, nval=100, th_res=th_res, min_omega=0.45, max_omega = max_omega, plot = True)
# dispQEP.CalculateBandStructure(nparam=5000, nval=500, th_res=th_res, min_omega=0.1, max_omega = max_omega, plot = True)
# plt.show()

omegas = np.linspace(min_omega, max_omega, nparam)
chern = dispQEP.ChernNumber_WLA(nzones=10, params=omegas, plot = True, buildRB=False, prefix='CN_III_')
print(chern)
plt.show()