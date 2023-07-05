import numpy as np
import matplotlib.pyplot as plt
import gif
import os

# Read input file for grid size and timesteps
with open('src/input_file.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith('Nx'):
            Nx = int(line.split(' ')[1])
        elif line.startswith('Ny'):
            Ny = int(line.split(' ')[1])
        elif line.startswith('timesteps'):
            timesteps = int(line.split(' ')[1])

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = ['solution_cpu.bin', 'solution_gpu.bin']
solution_files = [os.path.join(path, 'bin', file) for file in files]

# Save gif
@gif.frame
def plot_frame(frame, timestep, vmin=-1, vmax=1):
    plt.figure()
    plt.imshow(frame, vmin=vmin, vmax=vmax, cmap='coolwarm')
    cbar = plt.colorbar()
    cbar.set_label(r'Temperature, $T$ [K]')   
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(f'Timestep: {timestep}')
    plt.subplots_adjust(top = 0.95, bottom = .1, right = 1.0, left = -.25, hspace = 0, wspace = 0)
    if timestep == 0:
        plt.savefig(solution_file.replace('.bin', '_initial.png'))
    if timestep == timesteps - 1:
        plt.savefig(solution_file.replace('.bin', '_final.png'))


@gif.frame
def plot_frame_surface(frame, timestep, vmin=-1, vmax=1):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    ax.plot_surface(X, Y, frame, cmap='coolwarm', edgecolor='none', vmin=vmin, vmax=vmax)
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'Temperature, $T$ [K]')
    ax.view_init(30, 30)
    plt.title(f'Timestep: {timestep}')
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    if timestep == 0:
        plt.savefig(solution_file.replace('.bin', '_initial.png'))
    if timestep == timesteps - 1:
        plt.savefig(solution_file.replace('.bin', '_final.png'))


for solution_file in solution_files:
    print(f'Plotting {solution_file}...')
    try:
        solution = np.fromfile(solution_file, dtype=np.float32)
    except FileNotFoundError:
        print(f'File {solution_file} not found. Skipping...')
        continue
    try:
        solution = solution.reshape((timesteps, Nx, Ny))
    except ValueError:
        print(f'File {solution_file} has wrong dimensions. Skipping plotting...')
        continue

    vmin = np.min(solution)
    vmax = np.max(solution)

    # Save GIF
    frames = []
    for i in range(timesteps):
        if i % 10:
            continue
        frames.append(plot_frame(solution[i, :, :].T, i, vmin=vmin, vmax=vmax))
    gif.save(frames, solution_file.replace('.bin', '.gif'), duration=5)
