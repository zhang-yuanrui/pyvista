from ansys.dpf import core as dpf
from ansys.dpf.core import examples, vtk_helper

import deep_pixel_images as dpi

import vtk

import numpy as np

# from typing import List, Tuple

colorby_var_id = 1

def generate_deep_pixel_image(model: dpf.Model, var_field: dpf.Field):
    var_data = var_field.data
    is_scalar_data = var_data.ndim == 1
    var_unit = var_field.unit # strs
    print(f"unit: {var_unit}")
    var_name = var_field.name
    var_meshed_region = var_field.meshed_region
    dpf_unit_system = model.metadata.result_info.unit_system_name
    unit_system_to_name = dpf_unit_system.split(":", 1)[0]

    mats = var_meshed_region.property_field("mat") # property field
    
    grid = vtk_helper.dpf_mesh_to_vtk(var_meshed_region)
    grid = vtk_helper.append_field_to_grid(mats, var_meshed_region, grid, "Pick Data")
    grid = vtk_helper.append_field_to_grid(var_field, var_meshed_region, grid, var_name)

    # Create a vtkGeometryFilter to convert UnstructuredGrid to PolyData
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(grid)
    geometry_filter.Update()

    poly_data = geometry_filter.GetOutput()
    
    # a pyvista UnstructuredGrid is-a vtk UnstructuredGrid
    renderer, render_window= dpi.setup_render_routine(poly_data)
    rgb_buffer = dpi.get_rgb_value(render_window)
    pick_buffer = dpi.render_pick_data(grid, renderer, render_window)
    var_buffer = dpi.render_var_data(grid, renderer, render_window, var_name)
    
    global colorby_var_id # Todo: can be optimized in a class
    colorby_var_int = colorby_var_id
    colorby_var_id += 1
    colorby_var_decimal = 0 if is_scalar_data else 1 # Todo: .1, .2, .3 corresponds to x, y, z dimension. Only supports scalar for now
    colorby_var = f"{colorby_var_int}.{colorby_var_decimal}"
    
    # For now, it only supports one part with one variable
    json_data = {
        "parts": [
            {
                "name": "DPF sample", # hardcode
                "id": str(mats.data[0]), 
                "colorby_var": colorby_var
            }
        ],
        "variables": [
            {
                "name": var_name, 
                "id": str(mats.data[0]), 
                "pal_id": colorby_var_int,
                "unit_dims": "",
                "unit_system_to_name": unit_system_to_name,
                "unit_label": var_unit
            }
        ]
    }

    dpi.generate_tiff(json_data, rgb_buffer, pick_buffer, var_buffer, "dpf_sample_new.tiff")
    

def main():
    model = dpf.Model(examples.find_electric_therm())
    # mesh_info = model.metadata.mesh_info
    # print(f"body names: {mesh_info.body_names}")
    # print(f"zone names: {mesh_info.zone_names}")
    # print(f"part names: {mesh_info.part_names}")
    
    # print(model.metadata.result_info)
    # print(model.metadata.result_info[0].homogeneity)
    results = model.results
    electric_potential = results.electric_potential()
    fields = electric_potential.outputs.fields_container()
    print(f"fields: {fields}")
    potential = fields[0]
    
    
    generate_deep_pixel_image(model, potential)

if __name__ == "__main__":
    main()

def test():
    model = dpf.Model(examples.find_electric_therm())
    # model.plot()
    # model = dpf.Model(examples.find_simple_bar())
    results = model.results
    electric_potential = results.electric_potential()
    fields = electric_potential.outputs.fields_container()
    potential = fields[0]
    print(f"potential type: {type(potential)}")

    potential_data = potential.data # np.ndarray
    print(f"Potential data shape: {potential_data.shape}\nPotential data: {potential_data}")

    potential_unit = potential.unit # strs
    potential_name = potential.name
    print(f"potential name type: {type(potential_name)}")
    potential_meshed_region = potential.meshed_region

    mats = potential_meshed_region.property_field("mat") # property field

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
