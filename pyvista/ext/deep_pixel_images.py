import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

from PIL import Image, TiffImagePlugin
import json

# import report_utils

def create_sample_sphere():
    sphere = vtk.vtkSphereSource()
    print(type(sphere))
    sphere.Update()
    add_var_as_cell_data(sphere.GetOutput(), "Pick Data", lambda i: 3456)
    add_var_as_cell_data(sphere.GetOutput(), "Temperature", lambda i: i % 10000)

    return sphere

def create_unstructured_grid():
    # Create the points
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 0.0, 0.0)
    points.InsertNextPoint(1.0, 1.0, 0.0)
    points.InsertNextPoint(0.0, 1.0, 0.0)
    points.InsertNextPoint(0.0, 0.0, 1.0)
    points.InsertNextPoint(1.0, 0.0, 1.0)
    points.InsertNextPoint(1.0, 1.0, 1.0)
    points.InsertNextPoint(0.0, 1.0, 1.0)
    points.InsertNextPoint(2.0, 0.0, 0.0)
    points.InsertNextPoint(2.0, 1.0, 0.0)

    # Create an unstructured grid
    unstructured_grid = vtk.vtkUnstructuredGrid()
    unstructured_grid.SetPoints(points)

    # Create a hexahedron cell (cube)
    hexahedron = vtk.vtkHexahedron()
    for i in range(8):
        hexahedron.GetPointIds().SetId(i, i)  # Add the first 8 points to form a hexahedron

    # Create a tetrahedron cell
    tetra = vtk.vtkTetra()
    tetra.GetPointIds().SetId(0, 0)
    tetra.GetPointIds().SetId(1, 1)
    tetra.GetPointIds().SetId(2, 8)
    tetra.GetPointIds().SetId(3, 9)

    # Add the cells to the grid
    unstructured_grid.InsertNextCell(hexahedron.GetCellType(), hexahedron.GetPointIds())
    unstructured_grid.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())
    
    add_var_as_cell_data(unstructured_grid, "Pick Data", lambda i: 3456)
    add_var_as_point_data(unstructured_grid, "Temperature", lambda i: i % 10000)
    
    return unstructured_grid

def add_var_as_cell_data(poly_data, var_name, val_calculator):
    arr = vtk.vtkFloatArray()
    arr.SetName(var_name)
    arr.SetNumberOfComponents(1)
    num_cells = poly_data.GetNumberOfCells()
    arr.SetNumberOfTuples(num_cells)

    for i in range(num_cells):
        arr.SetValue(i, val_calculator(i))

    poly_data.GetCellData().AddArray(arr)

def add_var_as_point_data(poly_data, var_name, val_calculator):
    arr = vtk.vtkFloatArray()
    arr.SetName(var_name)
    arr.SetNumberOfComponents(1)
    num_points = poly_data.GetNumberOfPoints()
    arr.SetNumberOfTuples(num_points)

    for i in range(num_points):
        arr.SetValue(i, val_calculator(i))

    poly_data.GetPointData().AddArray(arr)
    # poly_data.GetOutput().GetPointData().SetActiveScalars(var_name)

def setup_render_routine(poly_data):
    # Mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Create the renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.SetMultiSamples(0)
    renderer.ResetCamera()
    render_window.AddRenderer(renderer)

    # render_window_interactor = vtk.vtkRenderWindowInteractor()
    # render_window_interactor.SetRenderWindow(render_window)

    renderer.AddActor(actor)

    return renderer, render_window #, render_windowdow_interactor

def get_vtk_scalar_mode(poly_data, var_name):
    point_data = poly_data.GetPointData()
    cell_data = poly_data.GetCellData()
    point_data_array = point_data.GetArray(var_name)
    cell_data_array = cell_data.GetArray(var_name)
    if point_data_array is not None:
        return vtk.VTK_SCALAR_MODE_USE_POINT_FIELD_DATA
    if cell_data_array is not None:
        return vtk.VTK_SCALAR_MODE_USE_CELL_FIELD_DATA
    raise ValueError(f"{var_name} does not belong to point data, nor cell data")

def setup_value_pass(poly_data, renderer, var_name):
    # Set up vtkValuePass
    value_pass = vtk.vtkValuePass()
    vtk_scalar_mode = get_vtk_scalar_mode(poly_data, var_name)
    value_pass.SetInputArrayToProcess(vtk_scalar_mode, var_name)
    value_pass.SetInputComponentToProcess(0)

    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(value_pass)  # Add value pass to the pass collection

    sequence = vtk.vtkSequencePass()
    sequence.SetPasses(passes)

    camera_pass = vtk.vtkCameraPass()
    camera_pass.SetDelegatePass(sequence)
    renderer.SetPass(camera_pass)

    return value_pass

def get_rgb_value(render_window):
    width, height = render_window.GetSize()
    print(f"Before: Render window size: {width}x{height}")
    # render_window.SetSize(800, 600)

    render_window.Render()
    
    width, height = render_window.GetSize()
    print(f"After: Render window size: {width}x{height}")

    # Capture the rendering result
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.Update()

    # Get the image data
    image_data = window_to_image_filter.GetOutput()

    # Convert VTK image data to a NumPy array
    width, height, _ = image_data.GetDimensions()
    vtk_array = image_data.GetPointData().GetScalars()
    np_array = vtk_to_numpy(vtk_array)

    # Reshape the array to a 3D array (height, width, 3) for RGB
    np_array = np_array.reshape(height, width, -1)
    # render_window_interactor.Start()

    return np_array

def render_pick_data(poly_data, renderer, render_window):
    value_pass = setup_value_pass(poly_data, renderer, "Pick Data")
    
    render_window.Render()

    buffer = value_pass.GetFloatImageDataArray(renderer)
    np_buffer = vtk_to_numpy(buffer)
    non_nan = np_buffer[~np.isnan(np_buffer)]
    width, height = render_window.GetSize()
    np_buffer = np_buffer.reshape(height, width)
    nan_mask = np.isnan(np_buffer)
    np_buffer = np.where(nan_mask, 0, np_buffer)
    np_buffer = np_buffer.astype(np.int16)
    pick_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    pick_buffer[:, :, 0] = np_buffer & 0xFF
    pick_buffer[:, :, 1] = (np_buffer >> 8) & 0xFF
    return pick_buffer

def render_var_data(poly_data, renderer, render_window, var_name):
    value_pass = setup_value_pass(poly_data, renderer, var_name)

    render_window.Render()

    buffer = value_pass.GetFloatImageDataArray(renderer)
    np_buffer = vtk_to_numpy(buffer)
    width, height = render_window.GetSize()
    np_buffer = np_buffer.reshape(height, width)
    return np_buffer

def generate_tiff(json_data, rgb_buffer, pick_buffer, var_buffer, output_file_name):
    image_description = json.dumps(json_data)

    rgb_image = Image.fromarray(rgb_buffer, mode='RGB')
    pick_image = Image.fromarray(pick_buffer, mode='RGBA')
    var_image = Image.fromarray(var_buffer, mode='F')

    tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
    tiffinfo[TiffImagePlugin.IMAGEDESCRIPTION] = image_description

    rgb_image.save(output_file_name, format='TIFF', save_all=True,
                    append_images=[pick_image, var_image], tiffinfo=tiffinfo)
    

def test():
    sphere = create_sample_sphere()
    renderer, render_window = setup_render_routine(sphere.GetOutput())
    rgb_buffer = get_rgb_value(render_window)
    print(rgb_buffer.shape)
    print("*************************")
    pick_buffer = render_pick_data(sphere.GetOutput(), renderer, render_window)
    print(f"Center of pick buffer: {pick_buffer[150][150]}")
    print("*************************")
    var_buffer = render_var_data(sphere.GetOutput(), renderer, render_window, "Temperature")
    print(var_buffer.shape)
    print(var_buffer[150][150])
    print("*************************")
    
    json_data = {
        "parts": [
            {
                "name": "sample", 
                "id": "3456", 
                "colorby_var": "3.0" 
            }
        ],
        "variables": [
            {
                "name": "Temperature", 
                "id": "3456", 
                "pal_id": "3", 
                "unit_dims": "M/LTT", 
                "unit_system_to_name": "SI", 
                "unit_label": "°C"
            }
        ]
    }

    generate_tiff(json_data, rgb_buffer, pick_buffer, var_buffer, "structured.tiff")

def main():
    test()

if __name__ == "__main__":
    main()