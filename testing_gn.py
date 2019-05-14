

import numpy as np

#true function: 4x^2 + 6x

A = np.array([[78.5367,1335.8479],[1335.8479,24559.9419]])

B = np.array([[-16.5888],[-300.8722]])

#print(np.dot(np.linalg.inv(A),B))

print(np.linalg.inv(A))

J = np.array([[1.284,1.6487,2.7183,3.4903,7.3891],
              [3.2101,8.2436,27.1828,43.6293,147.7811]])


print(np.dot(J,np.transpose(J)))