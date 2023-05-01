# CASE 2

import numpy as np
from fenics import *
from mshr import *
from mshr import Polygon, generate_mesh
import shapely.geometry as sg
import progressbar
import sys
import os
from helper import save_image_data, save_data
from testing_3channel import get_velocity_grid, get_pressure_grid, get_geometry_grid
import math
import matplotlib.pyplot as plt

################

resolution = 64
channel_length, channel_diameter, d1, d2, d3, x_mid, x1, y1, r1, x2, y2, r2 = 0.24, 5.00E-02,	4.00E-02,	2.00E-02,	2.00E-02,   1.29E-01,	2.22E-02,	1.08E-02,   5.15E-03,	3.28E-02,	3.97E-02,	4.54E-03

channel_length, channel_diameter, d1, d2, d3, x_mid, x1, y1, r1, x2, y2, r2 = 0.24, 5.00E-02,	4.00E-02,	2.00E-02,	2.00E-02,	9.83E-02,	2.58E-02,	1.13E-02,	3.19E-03,	5.47E-02,	4.00E-02,	4.55E-03


##################

geo = 2             #geometry2

res_data = []




# for automation
# case = float(sys.argv[1])

# resolution = float(sys.argv[2])
# # print(resolution)

# channel_diameter = float(sys.argv[3])

# d1, d2, d3 = float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
# # # print(channel_diameter)

# x_mid = float(sys.argv[7])

# x1, y1, r1 = float(sys.argv[9]), float(sys.argv[10]), float(sys.argv[11])


# x2, y2, r2 = float(sys.argv[12]), float(sys.argv[13]), float(sys.argv[14])


channel_length = 0.24             # cm
# x1, y1, r1 = 0.04, 0.010, 0.005
# x2, y2, r2 = 0.04, 0.040, 0.005
# d1 = 0.04
# d2 = 0.02
# d3 = 0.02
# x_mid = 0.18
# resolution = 64
bounds = [0, channel_length, 0, channel_diameter]

##########################new stuff##################



circle1 = sg.Point(x1, y1).buffer(r1)
circle2 = sg.Point(x2, y2).buffer(r2)

sg_poly = sg.Polygon([sg.Point(channel_length, (channel_diameter/2)-(d1/2), 0),
                sg.Point(channel_length, (channel_diameter/2)+(d1/2), 0),
                sg.Point(x_mid, (channel_diameter/2)+(d1/2), 0),
                sg.Point(x_mid-0.03, channel_diameter, 0),
                sg.Point(0.0, channel_diameter, 0),
                sg.Point(0.0, channel_diameter-d2, 0),
                sg.Point(x_mid-0.03, channel_diameter-d2, 0),
                sg.Point(x_mid, 0.025, 0),
                sg.Point(x_mid-0.03, d3, 0),
                sg.Point(0.0, d3, 0),
                sg.Point(0.0, 0.0, 0),
                sg.Point(x_mid-0.03, 0.0, 0),
                sg.Point(x_mid, (channel_diameter/2)-(d1/2), 0)])


sg_geometry = sg_poly.difference(circle1).difference(circle2)

####################################################


#TODO: Conditional statements for obstacles

T = 1.0             # final time
num_steps = 200     # number of time steps
dt = T / num_steps  # time step size
mu = 4              # kinematic viscosity
rho = 1.025         # density


cylinder1 = Circle(Point(x1, y1), r1)
cylinder2 = Circle(Point(x2, y2), r2)

poly = Polygon([Point(channel_length, (channel_diameter/2)-(d1/2), 0),
                Point(channel_length, (channel_diameter/2)+(d1/2), 0),
                Point(x_mid, (channel_diameter/2)+(d1/2), 0),
                Point(x_mid-0.03, channel_diameter, 0),
                Point(0.0, channel_diameter, 0),
                Point(0.0, channel_diameter-d2, 0),
                Point(x_mid-0.03, channel_diameter-d2, 0),
                Point(x_mid, 0.025, 0),
                Point(x_mid-0.03, d3, 0),
                Point(0.0, d3, 0),
                Point(0.0, 0.0, 0),
                Point(x_mid-0.03, 0.0, 0),
                Point(x_mid, (channel_diameter/2)-(d1/2), 0)])

domain = poly - cylinder1 - cylinder2
mesh = generate_mesh(domain, resolution) 

###################new stuff##############

# n_cells = mesh.num_cells()

###############################################

V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = f'near(x[0], %f)' %channel_length
cylinder1 = 'on_boundary && x[0]>{0} && x[0]<{1} && x[1]>{2} && x[1]<{3}'.format(x1-r1, x1+r1, y1-r1, y1+r1)
cylinder2 = 'on_boundary && x[0]>{0} && x[0]<{1} && x[1]>{2} && x[1]<{3}'.format(x2-r2, x2+r2, y2-r2, y2+r2)
walls = 'near(x[1],0) || near(x[1],{0}) || near(x[1],{1}) || near(x[1],{2}) || near(x[1],{3}) || near(x[1],{4})'.format(d3, channel_diameter-d2, channel_diameter, (channel_diameter/2)-(d1/2), (channel_diameter/2)+(d1/2))
walls1 = 'on_boundary && x[1]>0 && x[0]>{0} && x[1]<={1} && x[0]<={2}'.format(x_mid-0.03, channel_diameter, x_mid)

p_in = Expression("sin(t/5.0)", t=0.0, degree=2)
# p_in = Expression(f"0.0173 + 0.03 * sin (2 * {math.pi} * t)", t=0.0, degree=2)


# Define boundary conditions
bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
bcp_inflow  = DirichletBC(Q, p_in, inflow)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu_cylinder1 = DirichletBC(V, Constant((0, 0)), cylinder1)
bcu_cylinder2 = DirichletBC(V, Constant((0, 0)), cylinder2)
bcu_noslip1 = DirichletBC(V, Constant((0, 0)), walls1)
bcu = [bcu_noslip, bcu_noslip1, bcu_cylinder1, bcu_cylinder2]
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


if(resolution==32):
    img_size = 96, 16  
    folder = 'low'
else:
    img_size = 192, 32
    folder = 'high'

if not os.path.exists('results_hd5'):
    os.makedirs('results_hd5')

# # Create time series for saving solution for later
# timeseries_u = TimeSeries('results_hd5/{0}/geo{1}_case{2} velocity'.format(folder, geo, case))
# timeseries_p = TimeSeries('results_hd5/{0}/geo{1}_case{2} pressure'.format(folder, geo, case))


#Storing the coordinates of the unstructured mesh
x = V.tabulate_dof_coordinates().reshape(V.dim(), mesh.geometry().dim()) # gives coordinates of the nodes of the unstructured mesh
x_c = x[::2, :]     # x_coordinate
y_c = x[1::2, :]    # y_coordinate
coords_vel = [x_c[:, 0], y_c[:, 1]]
coords_vel = np.array(coords_vel).T

y = Q.tabulate_dof_coordinates().reshape(Q.dim(), mesh.geometry().dim()) # gives coordinates of the nodes of the unstructured mesh
coords_prr = np.array(y)


# Time-stepping
t = 0
for n in progressbar.progressbar(range(num_steps)):

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

    # #max velocity
    # print('max u1:', u_.vector().get_local().max())

    #Interpolation, etc
    u_n.assign(u_)
    p_n.assign(p_)
    
    # # Save solution to file (HDF5)
    # timeseries_u.store(u_.vector(), t)
    # timeseries_p.store(p_.vector(), t)

#     velocity_x, velocity_y = get_velocity_grid(u_, coords_vel, bounds, img_size, sg_geometry, savefig=False, showfig=False, top50=False)
#     pressure = get_pressure_grid(p_, coords_prr, bounds, img_size, sg_geometry, savefig=False, showfig=False, top50=False)

#     Z = np.stack([velocity_x, velocity_y, pressure], axis=2)
#     res_data.append(Z)

# if(resolution==32):
#   save_data(res_data, geo, int(case), folder)
# else:
#   save_data(res_data, geo, int(case), folder)

plt.savefig('poster_images/quiver/case2_1.png')
plt.show()
