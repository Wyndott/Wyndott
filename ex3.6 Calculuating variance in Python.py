# number of pets each person owns
data = [0, 1, 5, 7, 9, 10, 14]

def variances(values):
    mean = sum(values) / len(values)
    _variance = sum((v - mean) ** 2 for v in values) / len(values)
    return _variance
print (variances(data)) # prints 21.387755102040813
