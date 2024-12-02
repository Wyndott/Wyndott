import math
import pandas as pd

patient_data = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",").itertuples()

b0 = -3.17576395
b1 = 0.69267212

def logistic_function(x):
    p = 1.0 / (1.0 +math.exp(-(b0 + b1 * x)))
    return p

# Calculate the joint likelihood
joint_likelihood = 0.0

for p in patient_data:
   joint_likelihood = Sum(log((1.0 / (1.0 + exp(-(b + m * x(i)))))**y(i) * \
                              (1.0 - (1.0 / (1.0 + exp(-(b + m *x(i))))))**(1-y(i))), (i, 0, n))

joint_likelihood = math.exp(joint_likelihood)

print(joint_likelihood) # 4.7911180221699105e-05