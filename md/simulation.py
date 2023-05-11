from dataclasses import dataclass
from typing import Literal, Any
from collections import defaultdict
import numpy as np

from .system import *
from .integrator import *
from .forcefield import *

__all__ = ['Simulation']

@dataclass
class Simulation:
	Result = dict[str, list[Any]] 

	integ: Integrator
	ff: ForceField
	sys: System
	T: float  # temperature in Kelvin
	random_seed: int = 42

	def run(self, nsteps: int, accumulate: set[Literal['t', 'x', 'v', 'f', 'potential_total', 'potential_LJ', 'potential_elec', 'potential_bond', 'potential_bend', 'kinetic_energy', 'total_energy', 'temperature', 'pressure', 'virial']] = ['t', 'x']) -> 'Simulation.Result':
		self.sys.init(self.T, self.random_seed)
		self.integ.init()

		acc = defaultdict(list)

		if 'total_energy' in accumulate:
			accumulate.add('potential_total')
			accumulate.add('kinetic_energy')

		def log():
			if 't' in accumulate: acc['t'].append(self.integ.t)
			if 'x' in accumulate: acc['x'].append(self.sys.x.copy())
			if 'v' in accumulate: acc['v'].append(self.sys.v.copy())
			
			ff = self.integ.prev_ff if self.integ.prev_ff is not None else self.ff.forces(self.sys)
			if 'f' in accumulate: acc['f'].append(ff.forces.copy())
			if 'potential_total' in accumulate: acc['potential_total'].append(ff.potential_total)
			if 'potential_LJ' in accumulate: acc['potential_LJ'].append(ff.potential_LJ)
			if 'potential_elec' in accumulate: acc['potential_elec'].append(ff.potential_elec)
			if 'potential_bond' in accumulate: acc['potential_bond'].append(ff.potential_bond)
			if 'potential_bend' in accumulate: acc['potential_bend'].append(ff.potential_bend)
			
			if 'temperature' in accumulate: acc['temperature'].append(self.sys.temperature())
			if 'kinetic_energy' in accumulate: acc['kinetic_energy'].append(self.sys.kinetic_energy())
			if 'pressure' in accumulate: acc['pressure'].append(self.sys.pressure())
			if 'virial' in accumulate: raise NotImplementedError()

		def step():
			self.integ.step(self.sys, self.ff)

		log()

		for _ in range(nsteps):
			step()
			log()

		for key, val in acc.items():
			acc[key] = np.stack(val, axis=-1)

		if 'total_energy' in accumulate:
			acc['total_energy'] = acc['potential_total'] + acc['kinetic_energy'] 

		return dict(acc)  # convert to dict instead of defaultdict