''' Plot bayesian networks results when considering all nodes, not just the connected ones '''

import numpy as np
import matplotlib.pyplot as plt

net = [(1, 0.690000), (2, 0.690000), (2, 0.680000), (3, 0.690000), (4, 0.670000), (3, 0.680000), (3, 0.7000000)]
ill = [(2, 0.668421)]
ideo = [(1, 0.680000), (3, 0.680000), (4, 0.680000), (5, 0.680000), (6, 0.680000), (7, 0.680000), (4, 0.696000), (2, 0.672000), (3, 0.672000), (4, 0.672000), (5, 0.672000), (6, 0.672000), (7, 0.672000), (8, 0.672000)]

t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot([t[0] for t in net], [t[1] for t in net], 'r--', [t[0] for t in ill], [t[1] for t in ill], 'bs', [t[0] for t in ideo], [t[1] for t in ideo], 'g^')
plt.show()