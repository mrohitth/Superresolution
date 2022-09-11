"""
FEniCS tutorial demo program: Incompressible Navier-Stokes equations
for channel flow (Poisseuille) on the unit square using the
Incremental Pressure Correction Scheme (IPCS).

  u' + u . nabla(u)) - div(sigma(u, p)) = f
                                 div(u) = 0
"""

from __future__ import print_function
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *
from mshr import Polygon, generate_mesh
from scipy.interpolate import interp2d
import scipy.interpolate as _interp
import os

F=np.zeros((100,200))   #used for storing variable values
G=np.zeros((100,200))   #used for storing variable values


case = 1    #geometry1

#mesh2grid - function for intepolation 

def mesh2grid(v, mesh, n):
    """ Interpolates from an unstructured coordinates (mesh) to a structured
        coordinates (grid)
    """
    x = mesh[:, 0]  #x coordinates of the nodes of unstructured mesh  
    z = mesh[:, 1]  #y coordinates of the nodes of unstructured mesh  
    lx = x.max() - x.min()
    lz = z.max() - z.min()
    nn = v.size()   #v is the u_.vector() - stores all the velocity values at each node

    nx = 10     #number of points I want to divide x axis into, for the new structured mesh
    nz = 10     #number of points I want to divide y axis into, for the new structured mesh
    dx = lx/nx
    dz = lz/nz

    # construct structured grid
    x = np.linspace(x.min(), x.max(), nx)
    z = np.linspace(z.min(), z.max(), nz)
    X, Z = np.meshgrid(x, z)
    grid = stack(X.flatten(), Z.flatten()) #storing all of x and y coordinates of the nodes of new mesh 
    
    # interpolate to structured grid
    V = _interp.griddata(mesh, v, grid, 'cubic')

    # workaround edge issues
    if np.any(np.isnan(V)):
        W = _interp.griddata(mesh, v, grid, 'cubic')
        for i in np.where(np.isnan(V)):
            V[i] = W[i]                     #storing the velocity values at each node of the new mesh



    V = np.reshape(V, (nz, nx))
    V = np.reshape(V, (-1, 1))
    
    return V, nz, nx, grid
         
    # ss=10
    # newpath = r'/home/mathew/Desktop/new/rr%d' % ss
    # if not os.path.exists(newpath):
    #     os.makedirs(newpath)       
    # filename = ("rr%d/%.2di.csv" %(ss,n))
    # np.savetxt(filename , list(zip(grid[:,0], grid[:,1], V)), delimiter = ", ") #saving as a csv with (x, y, veloicity) as the 3 columns 
    # # return V, grid

def stack(*args):
    return np.column_stack(args)






T = 1.0             # final time
num_steps = 200     # number of time steps
dt = T / num_steps  # time step size
mu = 4              # kinematic viscosity
rho = 1.025         # density

channel = Rectangle(Point(0, 0), Point(2.4, 0.24))
box = Rectangle(Point(0.5, 0), Point(0.75, 0.1))
# box2 = Rectangle(Point(1.55, 0.1), Point(1.75, 0.24))
cylinder = Circle(Point(0.2, 0.12), 0.05)
cylinder2 = Circle(Point(1.2, 0.12), 0.05)

domain = channel - cylinder - cylinder2 - box 

mesh = generate_mesh(domain, 64) 

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.4)'
walls    = 'near(x[1], 0) || near(x[1], 0.24)'
# walls    = 'near(x[1], 0) || near(x[1], 0.1)'
box      = 'on_boundary && x[0]>=0.5 && x[1]>0 && x[0]<=0.75 && x[1]<=0.12'
# box2     = 'on_boundary && x[0]>=1.55 && x[1]>=0.1 && x[0]<=1.75 && x[1]<=0.24'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.05 && x[1]<0.25'
cylinder2 = 'on_boundary && x[0]>1.1 && x[0]<1.3 && x[1]>0.05 && x[1]<0.25'


p_in = Expression("sin(t/3.0)", t=0.0, degree=2)


# Define boundary conditions
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, p_in, inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
# bcu_inflow  = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_box = DirichletBC(V, Constant((0,0)), box)
# bcu_box2 = DirichletBC(V, Constant((0,0)), box2)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcu_cylinder2 = DirichletBC(V, Constant((0, 0)), cylinder2)
# bcu_noslip1 = DirichletBC(V, Constant((0, 0)), wall1)
bcu = [bcu_noslip, bcu_cylinder, bcu_box, bcu_cylinder2]
# bcu = [bcu_noslip, bcu_cylinder]
bcp = [bcp_inflow, bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)

# Define expressions used in variational forms
U   = 0.5*(u_n + u)
n   = FacetNormal(mesh)
f   = Constant((0, 0))
k   = Constant(dt)
mu  = Constant(mu)
rho = Constant(rho)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx + \
     rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create VTK files for visualization output
vtkfile_u = File('high/Case%d - straight2x0.5/velocity.pvd' %case)
vtkfile_p = File('high/Case%d - straight2x0.5/pressure.pvd' %case)

# Create time series for saving solution for later
# timeseries_u = TimeSeries('high/Case%d - straight2x0.5/velocity' %case)
# timeseries_p = TimeSeries('high/Case%d - straight2x0.5/pressure' %case)


# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    p_in.t = t

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    [bc.apply(b2) for bc in bcp]
    solve(A2, p_.vector(), b2)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3)

    # Plot solution
    plot(u_, title = 'Velocity')

    #max velocity
    print('max u:', u_.vector().get_local().max())
    
    # Save solution to file (VTK)
    vtkfile_u << (u_, t)
    vtkfile_p << (p_, t)

    # Save solution to file (HDF5)
    # timeseries_u.store(u_.vector(), t)
    # timeseries_p.store(p_.vector(), t)

    #Interpolation, etc
    u_n.assign(u_)
    p_n.assign(p_)

    x = V.tabulate_dof_coordinates().reshape(V.dim(), mesh.geometry().dim()) #gives coordinates of the nodes of the unstructured mesh
    x1 = x[:,0] #x values
    x2 = x[:,1] #y values
    L, nz, nx, grid = mesh2grid(u_.vector(), x, n)
    F[:,n] = L.ravel()

    y = Q.tabulate_dof_coordinates().reshape(Q.dim(), mesh.geometry().dim()) #gives coordinates of the nodes of the unstructured mesh
    y1 = y[:,0] #x values
    y2 = y[:,1] #y values
    M, nz, nx, grid = mesh2grid(p_.vector(), y, n)
    G[:,n] = M.ravel()

    # print(L)
    # print(F)

#saving
print(F.shape)  
print(G.shape)

newpath = r'/home/mathew/Desktop/new/testing/' 
if not os.path.exists(newpath):
    os.makedirs(newpath)       

np.savez("Case%d" %case, x = grid[:,0], y = grid[:,1], velocity = F, pressure = G)    



# Hold plot
# plt.savefig('images/foo1_4_64.png')
# plt.show()




