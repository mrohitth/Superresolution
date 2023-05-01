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


def get_velocity_grid(u, coords, bounds, img_size,  sg_geometry, savefig=False, showfig=False, top50=False):
  
  velocity = u.vector().get_local()
  velx = velocity[::2]
  vely = velocity[1::2]

  # Define the node coordinates and weights
  nodes = coords

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

  return Z1, Z2



def get_pressure_grid(p, coords, bounds, img_size,  sg_geometry, savefig=False, showfig=False, top50=False):
   
    prr = p.vector().get_local()

    # Define the node coordinates and weights
    nodes = coords
    weights = prr
    
    # Define the bounds of the rectangular domain and the resolution of the mesh
    xmin, xmax, ymin, ymax = bounds
    nx, ny = img_size

    # Create a mesh grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)

    grid = np.column_stack((X.flatten(), Y.flatten()))
    values = griddata(nodes, weights, grid, method='cubic')

    # Check which points lie inside or on the domain boundary
    mask = np.zeros_like(values)
    for i in range(grid.shape[0]):
        test_point = sg.Point(grid[i])
        if test_point.within(sg_geometry) or test_point.touches(sg_geometry):
            mask[i] = 1

    # Set values outside the domain to zero
    values *= mask

    # Check for nan values in the interpolated values
    Z = np.where(np.isnan(values), 0, values)

    # Reshape the values to match the mesh grid shape
    Z = values.reshape(X.shape)

    return Z



def get_geometry_grid(coords, sg_geometry, bounds, img_size):

    # Define the bounds of the rectangular domain and the resolution of the mesh
    xmin, xmax, ymin, ymax = bounds
    nx, ny = img_size

    # Create a mesh grid
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack((X.flatten(), Y.flatten()))

    # Check which points lie inside or on the domain boundary
    mask = np.zeros(grid.shape[0], dtype=bool)
    for i, node in enumerate(coords):
        if i < len(mask):
            test_point = sg.Point(node)
            if test_point.within(sg_geometry) or test_point.touches(sg_geometry):
                mask[i] = True

    # Reshape the mask to match the mesh grid shape
    mask = mask.reshape((ny, nx)).astype(int)

    return mask







