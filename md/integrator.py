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
from .forcefield import *

__all__ = ['Integrator', 'LeapFrog']

# Integrator

@dataclass
class Integrator:
	dt: float  # time step [ps]
	t: float = 0
	prev_ff: ForceField.Result | None = None

	def init(self):
		self.t = 0
		self.prev_ff = None

	def step(sys: System, ff: ForceField) -> None:
		raise NotImplementedError()
		# TODO : after stepping, remember to roll the particles back to the original box

@dataclass
class LeapFrog(Integrator):
	def step(self, sys: System, ff: ForceField):
		res = ff.forces(sys)
		sys.v += res.forces/sys.m[:, None] * self.dt/2
		sys.x += sys.v * self.dt
		res = ff.forces(sys)
		sys.v += res.forces/sys.m[:, None] * self.dt/2
		self.t += self.dt
		self.prev_ff = res