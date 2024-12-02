from sympy import *

from sympy.abc import x

def f(x):
    return x**2

def dx_f(x):
    return 2*x

slope_at_2 = dx_f(2.0)

print(slope_at_2) # prints 4.0

# calculate the slope at x = 2
print(dx_f.subs(x,2)) #prints 4