import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def load_grid_from_csv(filename):
    with open(filename, 'r') as f:
        layers = f.read().strip().split('\n\n')  # Split by empty lines
        grid = [np.loadtxt(layer.splitlines(), delimiter=',') for layer in layers]
    return np.array(grid)

def show_all_layers(filename):
    grid = load_grid_from_csv(filename)

    for i, layer in enumerate(grid):
        plt.figure()
        plt.title(f"Layer {i}")
        plt.imshow(layer, cmap='inferno', interpolation='nearest')
        plt.colorbar()
        plt.show()

def show_layer(filename, target_layer):
    grid = load_grid_from_csv(filename)

    for i, layer in enumerate(grid):
      if i == target_layer:
        plt.figure()
        plt.title(f"Layer {i}")
        plt.imshow(layer, cmap='inferno', interpolation='nearest')
        plt.colorbar()
        plt.show()

def show_3d_grid(filename):
    grid = load_grid_from_csv(filename)

    x, y, z = np.indices(grid.shape)

    fig = go.Figure(data=go.Volume(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=grid.flatten(),
        isomin=np.min(grid),
        isomax=np.max(grid),
        opacity=0.1,
        surface_count=20,
        colorscale="Inferno",
    ))

    fig.update_layout(scene=dict(
        xaxis=dict(title="X"),
        yaxis=dict(title="Y"),
        zaxis=dict(title="Z"),
    ))
    fig.show()