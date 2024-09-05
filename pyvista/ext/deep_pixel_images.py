# Get RGB
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

def create_sample_sphere():
    sphere = vtk.vtkSphereSource()

    add_var(sphere, "Pick Data", lambda i: 3456)
    add_var(sphere, "Temperature", lambda i: i % 10000)
    sphere.Update()

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


def setup_render_routine():
    # Create the renderer, render window, and interactor
    renderer = vtk.vtkRenderer()
    render_win = vtk.vtkRenderWindow()
    render_win.SetOffScreenRendering(1)
    render_win.SetMultiSamples(0)
    render_win.AddRenderer(renderer)

    # iRen = vtk.vtkRenderWindowInteractor()
    # iRen.SetRenderWindow(ren_win)
    
    return renderer, render_win

def render_rgb(geometry):
    renderer, render_win = setup_render_routine()

    # Mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(geometry.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0, 0, 0.4)
    
    renderer.AddActor(actor)

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
    print(np_array.shape)

    # Access the RGB values of any pixel
    x, y = 150, 150  # Example coordinates (mid point)
    rgb = np_array[y, x]
    print(f"Pixel at ({x}, {y}) has RGB values: {rgb}")

def main():
    sphere = create_sample_sphere()
    render_rgb(sphere)

# This block will only be executed if the script is run directly
if __name__ == "__main__":
    main()

