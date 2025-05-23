from sympy import *
from sympy.plotting import plot3d

def f(x):
    return x**2

def dx_f(x):
    return 2*x

#Declare x and y to SymPy
x,y = symbols('x y')

# Now just use Python syntax to declare function
f = 2*x**3 + 3*y**3

#Calculate the partial derivatives for x and y 
dx_f = diff(f, x)
dy_f = diff(f, y)

print(dx_f) # prints 6x**2
print(dy_f) # prints 9*y**2

# plot the function
plot3d(f)