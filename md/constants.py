from nptyping import NDArray, Shape, Float, UInt

__all__ = ['ATOM_TYPES', 'ATOM_MASSES', 'K_B', 'N_A', 'EPS0', 'ELEMENTARY_CHARGE']

# atom name -> atomic number
ATOM_TYPES: dict[str, UInt] = { 'H': 1, 'O': 16 }
# atomic masses, g/mol
ATOM_MASSES: dict[str, float] = { 'H': 1.0078, 'O': 15.9994 }

# https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/units.py
K_B = 8.314462159e-3  # boltzmann constant, kJ/(mol.K)
N_A = 6.02214076e23   # avogadro number [1/mol]
EPS0 = 5.526350e-3  # [e²/((kJ/mol).A)]
ELEMENTARY_CHARGE = 1.602176565e-19  # [As]