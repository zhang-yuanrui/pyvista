import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

def create_sample_sphere():
    sphere = vtk.vtkSphereSource()
    sphere.Update()
    add_var(sphere, "Pick Data", lambda i: 3456)
    add_var(sphere, "Temperature", lambda i: i % 10000)

    return sphere

def add_var(geometry, var_name, val_calculator):
    arr = vtk.vtkFloatArray()
    arr.SetName(var_name)
    arr.SetNumberOfComponents(1)
    num_points = geometry.GetOutput().GetNumberOfPoints()
    arr.SetNumberOfTuples(num_points)

    # Fill the pick data array with some dummy data
    for i in range(num_points):
        arr.SetValue(i, val_calculator(i))

    geometry.GetOutput().GetPointData().AddArray(arr)
    geometry.GetOutput().GetPointData().SetActiveScalars(var_name)


def setup_render_routine(geometry):
    # Create the renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_win = vtk.vtkRenderWindow()
    render_win.SetOffScreenRendering(1)
    render_win.SetMultiSamples(0)
    render_win.AddRenderer(renderer)

    # Mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(geometry.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0.4)
    
    renderer.AddActor(actor)
    
    return renderer, render_win

def setup_value_pass(renderer, var_name):
    # Set up vtkValuePass
    value_pass = vtk.vtkValuePass()
    value_pass.SetInputArrayToProcess(vtk.VTK_SCALAR_MODE_USE_POINT_FIELD_DATA, var_name)
    value_pass.SetInputComponentToProcess(0)

    passes = vtk.vtkRenderPassCollection()
    passes.AddItem(value_pass)  # Add value pass to the pass collection

    sequence = vtk.vtkSequencePass()
    sequence.SetPasses(passes)

    camera_pass = vtk.vtkCameraPass()
    camera_pass.SetDelegatePass(sequence)
    renderer.SetPass(camera_pass)

    return value_pass

def render_rgb(geometry):
    renderer, render_win = setup_render_routine(geometry)

    render_win.Render()

    # Capture the rendering result
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_win)
    window_to_image_filter.Update()

    # Get the image data
    image_data = window_to_image_filter.GetOutput()

    # Convert VTK image data to a NumPy array
    height, width, _ = image_data.GetDimensions()
    vtk_array = image_data.GetPointData().GetScalars()
    np_array = vtk_to_numpy(vtk_array)

    # Reshape the array to a 3D array (height, width, 3) for RGB
    np_array = np_array.reshape(height, width, -1)

    return np_array

def render_pick_data(geometry):
    renderer, render_win = setup_render_routine(geometry)
    value_pass = setup_value_pass(renderer, "Pick Data")
    
    render_win.Render()

    buffer = value_pass.GetFloatImageDataArray(renderer)
    np_buffer = vtk_to_numpy(buffer)
    width, height = render_win.GetSize()
    np_buffer = np_buffer.reshape(height, width)
    nan_mask = np.isnan(np_buffer)
    np_buffer = np.where(nan_mask, 0, np_buffer)
    np_buffer = np_buffer.astype(np.int16)
    pick_buffer = np.zeros((height, width, 4), dtype=np.uint8)
    pick_buffer[:, :, 0] = np_buffer & 0xFF
    pick_buffer[:, :, 1] = (np_buffer >> 8) & 0xFF 
    return pick_buffer

def render_var_data(geometry, var_name):
    renderer, render_win = setup_render_routine(geometry)
    value_pass = setup_value_pass(renderer, var_name)
    
    render_win.Render()

    buffer = value_pass.GetFloatImageDataArray(renderer)
    np_buffer = vtk_to_numpy(buffer)
    width, height = render_win.GetSize()
    np_buffer = np_buffer.reshape(height, width)
    return np_buffer

def main():
    sphere = create_sample_sphere()
    rgb_buffer = render_rgb(sphere)
    print(rgb_buffer.shape)
    print("*************************")
    pick_buffer = render_pick_data(sphere)
    print(pick_buffer[150][150])
    print("*************************")
    var_buffer = render_var_data(sphere, "Temperature")
    print(var_buffer[150][150])
    print("*************************")


if __name__ == "__main__":
    main()