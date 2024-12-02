import math

def predict_probablity(x, b0, b1):
    p = 1.0 / (1.- + math.exp(-(b0 + b1 * x)))
    return p