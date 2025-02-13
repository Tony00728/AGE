import numpy as np

weights = None
target_ages = 0 / 100
input_ages = 20 / 100
age_diffs = abs(target_ages - input_ages)

print(age_diffs)
weights = 0.5 * (np.sin(np.pi/2 * age_diffs)) + 2



print(weights)