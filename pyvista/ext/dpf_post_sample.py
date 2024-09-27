from ansys.dpf import core as dpf
from ansys.dpf.core import examples, vtk_helper

import deep_pixel_images as dpi

import vtk

import numpy as np


model = dpf.Model(examples.find_electric_therm())
# model.plot()
# model = dpf.Model(examples.find_simple_bar())
results = model.results
electric_potential = results.electric_potential()
fields = electric_potential.outputs.fields_container()
potential = fields[0]
print(f"potential type: {type(potential)}")
potential_dim = potential.dimensionality
print(f"dim type: {type(potential_dim)}\n dim: {potential_dim}")
potential_data = potential.data # np.ndarray
print(f"Potential data shape: {potential_data.shape}\nPotential data: {potential_data}")

potential_unit = potential.unit # strs
potential_name = potential.name
print(f"potential name type: {type(potential_name)}")
potential_meshed_region = potential.meshed_region

mats = potential_meshed_region.property_field("mat") # property field


# displacements = results.displacement()
# fields = displacements.outputs.fields_container()
# disp = fields[0]
# print(f"disp: {disp}")
# disp_data = disp.data # np.ndarray
# disp_data = disp_data.astype(np.float32)
# disp_unit = disp.unit # strs
# disp_name = disp.name
# disp_meshed_region = disp.meshed_region
# # print(f"disp mesh region: {disp_meshed_region}")

# mats = disp_meshed_region.property_field("mat") # property field

# arr = np.full(3000, 3456)
# mats.data = arr

# print(disp)
# print(mats)

grid = vtk_helper.dpf_mesh_to_vtk(potential_meshed_region)
grid = vtk_helper.append_field_to_grid(mats, potential_meshed_region, grid, "Pick Data")
grid = vtk_helper.append_field_to_grid(potential, potential_meshed_region, grid, "Potential")

# Create a vtkGeometryFilter to convert UnstructuredGrid to PolyData
geometry_filter = vtk.vtkGeometryFilter()
geometry_filter.SetInputData(grid)
geometry_filter.Update()

poly_data = geometry_filter.GetOutput()

# a pyvista UnstructuredGrid is-a vtk UnstructuredGrid
renderer, render_window= dpi.setup_render_routine(poly_data)
rgb_buffer = dpi.get_rgb_value(render_window)
# print(rgb_buffer[150][150])

pick_buffer = dpi.render_pick_data(grid, renderer, render_window)
# print(f"pick buffer shape: {pick_buffer.shape}\nSample in the center: {pick_buffer[150][150]}")

var_buffer = dpi.render_var_data(grid, renderer, render_window, "Potential")
# print(var_buffer[150][150])

json_data = {
    "parts": [
        {
            "name": "DPF sample", # hardcode
            "id": str(mats.data[0]), 
            "colorby_var": "3.0"  # hardcode
        }
    ],
    "variables": [
        {
            "name": potential_name, 
            "id": str(mats.data[0]), 
            "pal_id": "3", # hardcode
            "unit_dims": "M/LTT", # hardcode
            "unit_system_to_name": "SI", # hardcode
            "unit_label": potential_unit
        }
    ]
}

dpi.generate_tiff(json_data, rgb_buffer, pick_buffer, var_buffer, "dpf_sample.tiff")
