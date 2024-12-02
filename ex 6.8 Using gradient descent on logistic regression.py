from sympy import *
import pandas as pd

points = list(pd.read_csv("https://tinyurl.com/y2cocoo7").itertuples())

b1, b0, i, n = symbols('b1 b0 i n')
x, y = symbols('x y', cls=Function)

# Define the probability function
prob = 1 / (1 + exp(-(b0 + b1 * x(i))))

# Define the likelihood function
likelihood = prob**y(i) * (1 - prob)**(1 - y(i))

# Define the joint likelihood function
# Note: `n` should be the number of data points, and the range should be `n-1`
joint_likelihood = Sum(log(likelihood), (i, 0, n - 1))

# Partial derivative for m, with points substituted
d_b1 = diff(joint_likelihood, b1) \
                    .subs(n, len(points) - 1).doit() \
                    .replace(x, lambda i: points[i].x) \
                    .replace(y, lambda i: points[i].y)

# Partial derivative for m, with points substituted
d_b0 = diff(joint_likelihood, b0) \
                    .subs(n, len(points) - 1).doit() \
                    .replace(x, lambda i: points[i].x) \
                    .replace(y, lambda i: points[i].y)

# compile using lambdify for faster computation
d_b1 = lambdify([b1, b0], d_b1)
d_b0 = lambdify([b1, b0], d_b0)

# Perform Gradient Descent
b1 = 0.01
b0 = 0.01
L = .01

for j in range(10_000):
    b1 += d_b1(b1, b0) * L 
    b0 += d_b0(b1, b0) * L

print(b1, b0)
# 0.6767610419335931 -3.1110492213055765