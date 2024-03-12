import numpy as np

# All points are in format [cols, rows]
pt_A = [41, 2001]
pt_B = [2438, 2986]
pt_C = [3266, 371]
pt_D = [1772, 136]

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])

maxHeight = maxWidth = 450

input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
output_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])
print(input_pts)
print(output_pts)