from dataclasses import dataclass
import numpy as np
from nptyping import NDArray, Shape, Float, UInt
from pathlib import Path
import re
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
import itertools

from .constants import *
from .system import *
from .simulation import *

__all__ = [
	'plot_sys',
	'plot_xyz_trajectory',
	'plot_xy_trajectory',
	'plot_coord_trajectory',
	'plot_energy',
	'plot_rotating'
]

def plot_sys(sys: System):
	fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
	ax.set_proj_type('ortho')
	for i, j in sys.bonds:
		x1, x2 = sys.x[i], sys.x[j]
		d12 = sys.d12(i, j)
		ax.plot(*np.array((x1, x1+d12)).T, color='k', alpha=0.4)
	ax.scatter(*sys.x[sys.atomtypes == ATOM_TYPES['H']].T, color='blue', linewidth=0, alpha=1)
	ax.scatter(*sys.x[sys.atomtypes == ATOM_TYPES['O']].T, color='red', linewidth=0, alpha=1)
	ax.set_xlim((-sys.lattice[0]/2, +sys.lattice[0]/2))
	ax.set_ylim((-sys.lattice[1]/2, +sys.lattice[1]/2))
	ax.set_zlim((-sys.lattice[2]/2, +sys.lattice[2]/2))
	ax.set_title(f'System with $M={sys.N//3}$ water molecules')
	ax.set_xlabel('x [A]')
	ax.set_ylabel('y [A]')
	ax.set_zlabel('z [A]')
	return fig, ax

def plot_xyz_trajectory(res: Simulation.Result):
	assert 'x' in res

	fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
	ax.set_proj_type('ortho')
	ax.plot(res['x'][0, 0, :], res['x'][0, 1, :], res['x'][0, 2, :], color='red')
	ax.plot(res['x'][1, 0, :], res['x'][1, 1, :], res['x'][1, 2, :], color='blue')
	ax.plot(res['x'][2, 0, :], res['x'][2, 1, :], res['x'][2, 2, :], color='blue')
	ax.set_xlabel('x [A]')
	ax.set_ylabel('y [A]')
	ax.set_zlabel('z [A]')
	ax.set_title('xyz trajectory')

	return fig, ax

def plot_xy_trajectory(res: Simulation.Result):
	assert 'x' in res

	fig, ax = plt.subplots()
	ax.plot(res['x'][0, 0, :], res['x'][0, 1, :], label='O', color='red')
	ax.plot(res['x'][1, 0, :], res['x'][1, 1, :], label='H1', color='blue')
	ax.plot(res['x'][2, 0, :], res['x'][2, 1, :], label='H2', color='cyan')
	ax.legend()
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_title('xy trajectory')
	return fig, ax

def plot_coord_trajectory(res: Simulation.Result):
	assert 't' in res
	assert 'x' in res

	fig, axes = plt.subplots(ncols=3, nrows=3, sharex=True, layout='tight', figsize=(12, 9))
	for i, color, name in zip(range(3), ['red', 'blue', 'cyan'], ['O', 'H1', 'H2']):
		axes[0, i].set_title(name)
		for mu, coordname in zip(range(3), ['x', 'y', 'z']):
			axes[mu, i].plot(res['t'], res['x'][i, mu, :] - res['x'][i, mu, 0], color=color)
			axes[mu, i].set_xlabel('t [ps]')
			axes[mu, i].set_ylabel(f'{coordname}-{coordname}(0)')
	
	return fig, axes

def plot_energy(res: Simulation.Result):
	assert 't' in res
	assert 'potential_total' in res
	assert 'kinetic_energy' in res
	assert 'total_energy' in res
	assert 'potential_bond' in res
	assert 'potential_bend' in res
	assert 'potential_elec' in res

	fig, axes = plt.subplots(nrows=3, layout='tight', figsize=(9, 9))
	axes[0].plot(res['t'], res['potential_total'], color='blue', label='pot total')
	axes[0].plot(res['t'], res['kinetic_energy'], color='red', label='kin')
	axes[0].plot(res['t'], res['total_energy'], color='black', label='total')
	axes[0].set_title('energy [kJ/mol]')
	axes[0].legend(loc='upper left', bbox_to_anchor=(1,1))
	axes[1].plot(res['t'], res['potential_bond'], color='purple', label='pot bond')
	axes[1].plot(res['t'], res['potential_bend'], color='orange', label='pot bend')
	axes[1].plot(res['t'], res['potential_elec'], color='cyan', label='pot elec')
	axes[1].legend(loc='upper left', bbox_to_anchor=(1,1))
	axes[2].plot(res['t'], res['total_energy']-res['total_energy'][0])
	axes[2].set_xlabel('time [fs]')
	axes[2].set_ylabel('total energy drift [kJ/mol]')
	return fig, axes

def plot_rotating(sys: System, frames: int = 100, interval: int = 100):
	fig, ax = plot_sys(sys)
	azims = np.linspace(0, 360, frames)

	def animate(i):
		ax.view_init(azim=azims[i])
		return fig,

	ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)
	plt.close(fig)
	return HTML(ani.to_html5_video())