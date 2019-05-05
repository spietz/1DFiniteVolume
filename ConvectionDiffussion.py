
import simpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

print("1D Convection-Diffussion Problem")


## INPUT

bc = "dir-dir"
n = 10  # number of cells along x,y-axis
L = 1.0  # lenght of domain
Pe = 10.0  # global Peclet number
problem = 1.0  # problem to solve (1 or 2)
linsolver = "direct"  # linear solver ("direct","jacobi","gs","sor")
fvscheme = "cds-cds"  # finite volume scheme ("cds-cds" or "uds-cds")
omegaSOR = 1.93  # relaxation parameter for SOR (0<=omegaSOR<2)
imax = 10000  # maximum number of iterations for linsolver
tol = 1e-12  # relative tolerance for linsolver


## EXACT SOLUTION

if problem == 1:
        def temp_exact(Pe, x):
                return (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
elif problem == 2:
        def temp_exact(Pe, x):
                return (np.exp(Pe*x)-1)/(np.exp(Pe)-1)
else:
        print("problem not implemented")

temp_a = 0  # Temperature western bc of domain
temp_b = 1  # Temperature eastern bc of domain

# Flux through western bc of domain
flux_a = (Pe*np.exp(Pe*0))/(np.exp(Pe) - 1)

# Flux through eastern bc of domain
flux_b = (Pe*np.exp(Pe*1))/(np.exp(Pe) - 1)


## PRE-PROCESSING

# Coordinate arrays
dx = L/n  # cell size in x,y-direction

## Generate velocity field
u_w = np.ones(n)  # western face velocity, u = 1
u_e = np.ones(n)  # eastern face velocity, u = 1 source term
s = np.zeros(n)

# Generate convective face flux matrices
conv_face_flux_w = Pe*u_w
conv_face_flux_e = Pe*u_e

# Generate coefficient matrices
if fvscheme.lower() == "cds-cds":  # CDS-CDS FV-scheme applied

        if(dx*Pe >= 2):  # Bad diagonal dominance
                print("warning: dx*Pe>=2")

        # obtained from (5.24)/(5.22)
        aw = (-1./dx)-conv_face_flux_w/2
        ae = (-1./dx)+conv_face_flux_e/2
        ap = -(aw+ae)-conv_face_flux_w+conv_face_flux_e

elif fvscheme.lower() == "uds-cds":  # UDS-CDS FV-scheme applied

        # obtained from (5.27)/(5.22)
        aw = (-1./dx)+np.minimum(0, -conv_face_flux_w)
        ae = (-1./dx)+np.minimum(0, conv_face_flux_e)
        ap = -(ae+aw)+conv_face_flux_e-conv_face_flux_w

else:    # fvscheme unknown
        print("fvscheme not implemented")


# Impose boundary conditions and compute rhs vector from BC
if fvscheme.lower() == "cds-cds":  # CDS-CDS FV-scheme applied

        if bc.lower() == "dir-dir":  # specified temp_a, temp_b

                # Dirichlet: 2nd order polynomial (5.44)
                ap[0] = ap[0]-2*aw[0]
                ae[0] = ae[0]+(1./3.)*aw[0]
                s[0] = s[0]-(8./3.)*aw[0]*temp_a
                aw[0] = 0

                # Dirichlet: 2nd order polynomial (5.45)
                ap[n-1] = ap[n-1]-2*ae[n-1]
                aw[n-1] = aw[n-1]+(1./3.)*ae[n-1]
                s[n-1] = s[n-1]-(8./3.)*ae[n-1]*temp_b
                ae[n-1] = 0

        elif bc.lower() == "dir-neu":  # specified temp_a, flux_b

                # Dirichlet: 2nd order polynomial (5.44)
                ap[0] = ap[0]-2*aw[0]
                ae[0] = ae[0]+(1./3.)*aw[0]
                s[0] = s[0]-(8./3.)*aw[0]*temp_a
                aw[0] = 0

                # Neumann: 2nd order FD
                ap[n-1] = ap[n-1]+ae[n-1]
                s[n-1] = s[n-1] - ae[n-1]*dx*flux_b
                ae[n-1] = 0

        elif bc.lower() == "neu-dir":  # specified flux_a, temp_b

                # Neumann: 2nd order FD
                ap[0] = ap[0]+aw[0]
                s[0] = s[0]+aw[0]*dx*flux_a
                aw[0] = 0

                # Dirichlet: 2nd order polynomial (5.45)
                ap[n-1] = ap[n-1]-2*ae[n-1]
                aw[n-1] = aw[n-1]+(1./3.)*ae[n-1]
                s[n-1] = s[n-1]-(8./3.)*ae[n-1]*temp_b
                ae[n-1] = 0

elif fvscheme.lower() == "uds-cds":  # UDS-CDS FV-scheme applied

        if bc.lower() == "dir-dir":  # specified temp_a, temp_b

                # Dirichlet: 1st order polynomial (5.40)
                ap[0] = ap[0]-aw[0]
                s[0] = s[0]-aw[0]*2*temp_a
                aw[0] = 0

                # Dirichlet: 1st order polynomial (5.41)
                ap[n-1] = ap[n-1]-ae[n-1]
                s[n-1] = s[n-1]-ae[n-1]*2*temp_b
                ae[n-1] = 0

        elif bc.lower() == "dir-neu":  # specified temp_a, flux_b

                # Dirichlet: 1st order polynomial (5.40)
                ap[0] = ap[0]-aw[0]
                s[0] = s[0]-aw[0]*2*temp_a
                aw[0] = 0

                # Neumann: 2nd order FD
                ap[n-1] = ap[n-1]+ae[n-1]
                s[n-1] = s[n-1] - ae[n-1]*dx*flux_b
                ae[n-1] = 0

        elif bc.lower() == "neu-dir":  # specified flux_a, temp_b

                # Neumann: 2nd order FD
                ap[0] = ap[0]+aw[0]
                s[0] = s[0]+aw[0]*dx*flux_a
                aw[0] = 0

                # Dirichlet: 1st order polynomial (5.41)
                ap[n-1] = ap[n-1]-ae[n-1]
                s[n-1] = s[n-1]-ae[n-1]*2*temp_b
                ae[n-1] = 0

else:    # fvscheme unknown
        print("fvscheme not implemented")

## Assemble tri-diagonal system matrix
data = np.zeros((3, n))
data[0, 0:n-1] = aw[1:n]
data[1, :] = ap
data[2, 1:n] = ae[0:n-1]
offsets = np.array([-1, 0, 1])
A = spdiags(data, offsets, n, n, format="csc")


## Solve system of linear equations

# Solve equations using either direct or iterative method
if linsolver.lower() == "direct":  # direct solver
        temp = spsolve(A, s)
elif linsolver.lower() == "jacobi" \
     or linsolver.lower() == "gs" or linsolver.lower() == "sor":
        temp, _, _, _, _ = simpy.solve(A, s,
            "sor", 500, 1e-4, omegaSOR, s*0, False)  # iterative solution
else:
        print("linsolver not implemented")


## POST-PROCESSING

# Extrapolate temperature field to walls
ttemp = np.zeros(n+2)  # initialize extended temperature array
ttemp[1:n+1:1] = temp[:]  # cells temperatures

# Compute temperature gradients at walls
if fvscheme.lower() == "cds-cds":  # CDS-CDS FV-scheme applied

        if bc.lower() == "dir-dir":  # specified temp_a, temp_b
                ttemp[0] = ((4./3.)*temp_a-(1./2.)*temp[0]+(1./6.)*temp[1])
                ttemp[n+1] = ((4./3.)*temp_b-(1./2.)*temp[n-1]+(1./6.)*temp[n-2])

        elif bc.lower() == "dir-neu":  # specified temp_a, flux_b
                ttemp[0] = ((4./3.)*temp_a-(1./2.)*temp[0]+(1./6.)*temp[1])
                ttemp[n+1] = temp[n-1]+0.5*dx*flux_b

        elif bc.lower() == "neu-dir":  # specified flux_a, temp_b
                ttemp[0] = temp[0]-0.5*dx*flux_a
                ttemp[n+1] = ((4./3.)*temp_b-(1./2.)*temp[n-1]+(1./6.)*temp[n-2])

elif fvscheme.lower() == "uds-cds":  # UDS-CDS FV-scheme applied

        if bc.lower() == "dir-dir":  # specified temp_a, temp_b
                # t correct because the scheme assumes tw ~ t{i-1}, te ~ t{i}
                ttemp[0] = 2*temp_a-temp[0]
                ttemp[n+1] = temp[n-1]

        elif bc.lower() == "dir-neu":  # specified temp_a, flux_b
                ttemp[0] = 2*temp_a-temp[0]
                ttemp[n+1] = temp[n-1]+0.5*dx*flux_b

        elif bc.lower() == "neu-dir":  # specified flux_a, temp_b
                ttemp[0] = temp[0]-0.5*dx*flux_a
                ttemp[n+1] = temp[n-1]


# Convective and diffusive wall fluxes, global cons. of heat, notes eq. (5.47)
conv_flux_a = conv_face_flux_w[0]*ttemp[0]
conv_flux_b = conv_face_flux_e[n-1]*ttemp[n+1]
if fvscheme.lower() == "cds-cds":  # CDS-CDS FV-scheme applied

        if bc.lower() == "dir-dir":  # specified temp_a, temp_b
                diff_flux_a = 1./(3*dx)*(-8*temp_a+9*temp[0]-temp[1])
                diff_flux_b = 1./(3*dx)*(temp[n-2]-9*temp[n-1]+8*temp_b)

        elif bc.lower() == "dir-neu":  # specified temp_a, flux_b
                diff_flux_a = 1./(3*dx)*(-8*temp_a+9*temp[0]-temp[1])
                diff_flux_b = flux_b

        elif bc.lower() == "neu-dir":  # specified flux_a, temp_b
                diff_flux_a = flux_a
                diff_flux_b = 1./(3*dx)*(temp[n-2]-9*temp[n-1]+8*temp_b)

elif fvscheme.lower() == "uds-cds":  # UDS-CDS FV-scheme applied

        if bc.lower() == "dir-dir":  # specified temp_a, temp_b
                diff_flux_a = 2./dx*(temp[0]-temp_a)
                diff_flux_b = 2./dx*(temp_b-temp[n-1])

        elif bc.lower() == "dir-neu":  # specified temp_a, flux_b
                diff_flux_a = 2./dx*(temp[0]-temp_a)
                diff_flux_b = flux_b

        elif bc.lower() == "neu-dir":  # specified flux_a, temp_b
                diff_flux_a = flux_a
                diff_flux_b = 2./dx*(temp_b-temp[n-1])

# Calculate flux error
flux_error = abs((conv_flux_a-diff_flux_a)-(conv_flux_b-diff_flux_b))

## PLOTS

plt.ion()

# Coordinate arrays
dx = L/n  # cell size in x,y-direction
xc = np.arange(dx/2., L, dx)  # cell center coordinate vector along x,y-axis
xt = np.zeros(n+2)
xt[n+1] = 1
xt[1:n+1:1] = xc[:]  # extended cell center coor. vector, incl. walls

## Compute exact solution
temp_ex = temp_exact(Pe, xt)  # exact solution for given problem,Pe,n


# Temperature field
f, axarr = plt.subplots(2, sharex=True)

axarr[0].set_title("Convection-diffusion by %s for Pe = %d \n Flux-error = %0.3e" \
           % (fvscheme, Pe, flux_error))
axarr[0].plot(xt[:], ttemp[:], '-ko', linewidth=2, label="numerical")
axarr[0].plot(xt[:], temp_ex[:], '--kx', linewidth=2, label="analytical")
axarr[0].legend(loc=0)
axarr[0].grid(True)
axarr[0].set_xlabel("x")
axarr[0].set_ylabel("T")

# Error
axarr[1].plot(xt[:], ttemp[:]-temp_ex[:], '-k*')
axarr[1].set_xlabel("x")
axarr[1].set_ylabel("error")
axarr[1].set_title("Truncation error on T(x)")
axarr[1].grid(True)

plt.tight_layout()

print("done!")
