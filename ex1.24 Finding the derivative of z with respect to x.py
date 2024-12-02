from sympy import *
from sympy import diff
from sympy.abc import x
z = (x**2 + 1)**3 - 2
dz_dx = diff(z, x)
print(dz_dx)

# 6*x*(x**2 + 1)**2
