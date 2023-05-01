import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from fenics import *
from mshr import *
from mshr import Polygon, generate_mesh
import shapely.geometry as sg
import sys
import progressbar


def save_image_data(u, coords, t, bounds, img_size,  sg_geometry, savefig=False, showfig=False, top50=False):
  
  velocity = u.vector().get_local()
  velx = velocity[::2]
  vely = velocity[1::2]
  # d = np.column_stack((velx, vely))
  # vel_mag = np.linalg.norm(d, axis=1)

  # Define the node coordinates and weights
  nodes = coords
  # weights = vel_mag

  # Define the bounds of the rectangular domain and the resolution of the mesh
  xmin, xmax, ymin, ymax = bounds
  nx, ny = img_size

  # Create a mesh grid
  x = np.linspace(xmin, xmax, nx)
  y = np.linspace(ymin, ymax, ny)
  X, Y = np.meshgrid(x, y)
  
  grid = np.column_stack((X.flatten(), Y.flatten()))
  values_x = griddata(nodes, velx, grid, method='cubic')
  values_y = griddata(nodes, vely, grid, method='cubic')


  # Check which points lie inside the domain
  mask_x = np.zeros_like(values_x)
  mask_y = np.zeros_like(values_y)

  for i in range(grid.shape[0]):
      test_point = sg.Point(grid[i])
      if test_point.within(sg_geometry):
          mask_x[i] = 1
          mask_y[i] = 1

  # Set values outside the domain to zero
  values_x *= mask_x
  values_y *= mask_y

  # Check for nan values in the interpolated values
  Z1 = np.where(np.isnan(values_x), 0, values_x)
  Z2 = np.where(np.isnan(values_y), 0, values_y)

  # Reshape the values to match the mesh grid shape
  Z1 = values_x.reshape(X.shape)
  Z2 = values_y.reshape(X.shape)

  # Display the resulting image
  if showfig or savefig:
    fig, ax = plt.subplots()
    im = ax.imshow(Z1, cmap='BuGn', origin='lower', extent=[xmin, xmax, ymin, ymax])   

    if top50:
      # Get indices of top 50 maximum velocities
      top50_indices = np.argsort(velx)[-50:]

      # Get coordinates of nodes corresponding to top 50 maximum velocities
      top50_nodes = nodes[top50_indices]

      # Create new plot of the same mesh grid, with different marker shape and color for top 50 nodes
      ax.scatter(top50_nodes[:, 0], top50_nodes[:, 1], c='r', marker='o', s=50)
    
    # Add a color bar to show the weight values
    cbar = fig.colorbar(im)
    cbar.set_label('Velocity')

    # Add a title and axis labels
    ax.set_title(f'Velocity x at t: {t:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if savefig:
        if not os.path.exists('results'):
              os.makedirs('results')
        plt.savefig(f"results/velocity_x_t_{t:.2f}.png")

    # Show the plot
    if showfig:
      plt.show()
    else:
      plt.close()

  return Z1, Z2




def save_data(res_data, geo, case, name):

  res_data = np.array(res_data)

  # Change all NaN values to 0
  res_data[np.isnan(res_data)] = 0
  
  if not os.path.exists(f'results_data_generated/{name}'):
    os.makedirs(f'results_data_generated/{name}')

  np.savez(f"results_data_generated/{name}/geo{geo}_case{case}", image_data = res_data)