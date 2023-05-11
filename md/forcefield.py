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

__all__ = ['ForceField', 'FBAeps']

# Force fields

@dataclass
class ForceField:
	class Result:
		def __init__(self, N: int):
			self.forces: NDArray[Shape['N,3'], Float] = np.zeros((N, 3), dtype=Float)  # [kJ/(mol.A)]
			self.potential_total: float = 0.0  # [kJ/mol]
			self.potential_LJ: float = 0.0  # [kJ/mol]
			self.potential_elec: float = 0.0  # [kJ/mol]
			self.potential_bond: float = 0.0  # [kJ/mol]
			self.potential_bend: float = 0.0  # [kJ/mol]

	def forces(self, sys: System) -> 'ForceField.Result':
		raise NotImplementedError()

@dataclass
class FBAeps(ForceField):
	eps_OO: float = 0.792324   # O-O Lennard-Jones energy scale [kJ/mol]
	sigma_OO: float = 3.1776   # O-O Lennard-Jones repulsive diameter pair [A]
	q_O: float = -0.8450       # O site charge [e]
	q_H: float = 0.4225        # H site charge [e]
	r_OH: float = 1.027        # O-H bond resting distance [A]
	k_OH: float = 3000         # O-H bond spring coefficient [kJ/(mol.A²)]
	theta_HOH: float = np.deg2rad(114.70)  # H-O-H resting angle [rad]
	k_HOH: float = 383         # H-O-H angle spring coefficient [kJ/(mol.rad²)]

	def forces(self, sys: System) -> ForceField.Result:
		res = ForceField.Result(N=len(sys.x))
		self.forces_LJ(sys, res)
		self.forces_elec(sys, res)
		self.forces_bond(sys, res)
		self.forces_bend(sys, res)
		return res

	def forces_LJ(self, sys: System, res: ForceField.Result | None = None) -> ForceField.Result:
		if res is None:
			res = ForceField.Result(N=len(sys.x))

		for a, b in itertools.product(range(sys.N), range(sys.N)):
			# TODO : precompute these indices
			if a == b: continue
			if sys.atomtypes[a] != ATOM_TYPES['O'] or sys.atomtypes[b] != ATOM_TYPES['O']: continue
			
			d = sys.d12(a, b)
			r = np.linalg.norm(d)
			f = -4*self.eps_OO * (12*self.sigma_OO**12/r**13 - 6*self.sigma_OO**6/r**7) * d/r
			res.forces[a] += f
			res.forces[b] += -f
			res.potential_LJ += 4*self.eps_OO * (self.sigma_OO**12/r**12 - self.sigma_OO**6/r**6)
		
		res.potential_total += res.potential_LJ
		return res

	def forces_elec(self, sys: System, res: ForceField.Result | None = None) -> ForceField.Result:
		if res is None:
			res = ForceField.Result(N=len(sys.x))

		for a, b in itertools.product(range(sys.N), range(sys.N)):
			# TODO : precompute these indices
			if a == b: continue

			# TODO : precompute this
			q_a = self.q_O if sys.atomtypes[a] == ATOM_TYPES['O'] else self.q_H
			q_b = self.q_O if sys.atomtypes[b] == ATOM_TYPES['O'] else self.q_H

			d = sys.d12(a, b)
			r = np.linalg.norm(d)
			f = -1/(4*np.pi*EPS0) * q_a*q_b/r**2 * d/r

			res.forces[a] += f
			res.forces[b] += -f
			res.potential_elec += 1/(4*np.pi*EPS0) * q_a*q_b/r

		res.potential_total += res.potential_elec
		return res

	def forces_bond(self, sys: System, res: ForceField.Result | None = None) -> ForceField.Result:
		if res is None:
			res = ForceField.Result(N=len(sys.x))

		for a, b in sys.bonds:
			d = sys.d12(a, b)
			r = np.linalg.norm(d)
			f = self.k_OH * (r - self.r_OH) * d/r

			res.forces[a] += f
			res.forces[b] += -f
			res.potential_bond += self.k_OH/2 * (r - self.r_OH)**2

		res.potential_total += res.potential_bond
		return res

	def forces_bend(self, sys: System, res: ForceField.Result | None = None) -> ForceField.Result:
		# https://math.stackexchange.com/questions/1165532/gradient-of-an-angle-in-terms-of-the-vertices
		
		if res is None:
			res = ForceField.Result(N=len(sys.x))

		for a, b, c in sys.angles:
			dOH1 = sys.d12(b, a)
			dOH2 = sys.d12(b, c)
			rOH1 = np.linalg.norm(dOH1)
			rOH2 = np.linalg.norm(dOH2)
			theta = np.arccos(np.dot(dOH1, dOH2) / (rOH1 * rOH2))
			
			num1 = np.cross(dOH1, np.cross(dOH1, dOH2))
			dtheta1 = num1 / np.linalg.norm(num1) / rOH1
			fH1 = -self.k_HOH * (theta - self.theta_HOH) * dtheta1
			res.forces[a] += fH1
			
			num2 = np.cross(dOH2, np.cross(dOH2, dOH1))
			dtheta2 = num2 / np.linalg.norm(num2) / rOH2
			fH2 = -self.k_HOH * (theta - self.theta_HOH) * dtheta2
			res.forces[c] += fH2

			res.forces[b] += -(fH1 + fH2)

			res.potential_bend += self.k_HOH/2 * (theta - self.theta_HOH)**2

		res.potential_total += res.potential_bend
		return res