from dataclasses import dataclass
import numpy as np
from nptyping import NDArray, Shape, Float, UInt
from pathlib import Path
import re
from .constants import *

__all__ = ['System']

@dataclass
class System:
	x: NDArray[Shape['N,3'], Float]           # positions [A]
	v: NDArray[Shape['N,3'], Float]           # velocities [A/ps]
	m: NDArray[Shape['N,3'], Float]           # masses [atomic units]
	lattice: NDArray[Shape['3'], Float]       # dimensions of the periodic cubic lattice [A]
	bonds: NDArray[Shape['B,2'], UInt]        # bond indices
	angles: NDArray[Shape['A,3'], UInt]       # angle indices
	atomtypes: NDArray[Shape['N'], UInt]      # atom types

	def __post_init__(self):
		assert len(self.x) == len(self.v) == len(self.m) == len(self.atomtypes)
		self.N = len(self.x)
		# TODO : precompute all pair indices
		# TODO : precompute all OO indices

	def init(self, T: float, random_seed: int = 42):
		"""Initialize the system

		Samples momenta from the Maxwell-Boltzmann distribution (see https://manual.gromacs.org/documentation/current/reference-manual/algorithms/molecular-dynamics.html#coordinates-and-velocities)

		Parameters
		----------
		T : float
			temperature in Kelvin
		random_seed : int, optional
			random seed, by default 42
		"""
		self.v = np.random.default_rng(random_seed).normal(0, np.sqrt(T*K_B/self.m[:, None]), (self.N, 3))

	def d12(self, i: int, j: int) -> NDArray[Shape['3'], Float]:
		"""Compute the distance vector 1->2 between ``x1`` and ``x2`` in the cubic lattice"""
		x1, x2 = self.x[i], self.x[j]
		d = np.array((x2 - x1, x2 - x1 + self.lattice, x2 - x1 - self.lattice))
		return d[np.argmin(np.abs(d), axis=0), [0,1,2]]

	def temperature(self) -> float:
		"""temperature of the system in K"""
		return 2*self.kinetic_energy() / (3*self.N*K_B)

	def kinetic_energy(self) -> float:
		"""kinetic energy in kJ/mol"""
		return np.sum(np.sum(self.v**2, axis=1)*self.m)/2

	def volume(self) -> float:
		"""volume of the simulation box in A³"""
		return np.prod(self.lattice)

	def density(self) -> float:
		"""density in 1/A³"""
		return self.m.sum() / self.volume()

	def pressure(self) -> float:
		"""pressure in kJ/(mol.A³)"""
		raise NotImplementedError()
		# TODO : the virial must be computed from the forces
		# return (2*self.kinetic_energy() + virial)/(3*self.volume())

	@staticmethod
	def load_NIST(filepath: Path) -> 'System':
		with open(filepath, 'r') as file:
			lattice = np.array([float(l) for l in re.split(r'\s+', file.readline().strip()) ])
			A = int(file.readline().strip())  # A : number of molecules (1 angle per molecule)
			B = 2*A  # B : number of bonds (2 bonds per molecule)
			N = 3*A  # N : number of atoms (3 atoms per molecule)

			x = np.zeros((N, 3), dtype=Float)
			v = np.zeros((N, 3), dtype=Float)
			m = np.zeros(N, dtype=Float)
			atomtypes = np.zeros(N, dtype=UInt)
			bonds = np.zeros((B, 2), dtype=UInt)
			angles = np.zeros((A,  3), dtype=UInt)
			i = 0
			b = 0
			for a in range(A):
				bonds[b, :]   = i, i+1
				bonds[b+1, :] = i, i+2
				angles[a, :]  = i+1, i, i+2
				for _ in range(3):
					line = re.split(r'\s+', file.readline().strip())
					x[i, :] = [ float(x_) for x_ in line[1:4] ]
					atomtypes[i] = ATOM_TYPES[line[4]]
					m[i] = ATOM_MASSES[line[4]]
					assert i == int(line[0])-1
					i += 1
				b += 2

			return System(x=x, v=v, m=m, lattice=lattice, bonds=bonds, angles=angles, atomtypes=atomtypes)

	def init_maxwellboltzmann(self):
		# TODO : sample velocities from maxwell-boltzmann
		raise NotImplementedError()


def __main__():
	from .plotting import plot_sys

	sys = System.load_NIST('data/nist_SPC_E_water_data/spce_sample_config_periodic1.txt')
	sys.init(T=25+273.15)
	plot_sys(sys);