''' Plot bayesian networks results when considering all nodes, not just the connected ones '''

import numpy as np
import matplotlib.pyplot as plt

net = [(0, 0.607143), (1, 0.607143), (2, 0.607143), (3, 0.607143), (4, 0.607143)]
ill = [(4, 0.661538), (5, 0.661538), (5, 0.692308)]
ideo = [(0, 0.707692), (1, 0.707692), (2, 0.707692), (3, 0.707692), (4, 0.707692), (5, 0.707692), (1, 0.730769), (2, 0.730769), (3, 0.730769), (4, 0.730769), (5, 0.730769), (1, 0.715385), (2, 0.715385), (3, 0.715385), (4, 0.715385), (5, 0.715385), (2, 0.723077), (3, 0.723077), (4, 0.723077), (5, 0.723077), (1, 0.746154), (2, 0.746154), (3, 0.746154), (4, 0.746154), (5, 0.746154)]

t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot([t[0] for t in net], [t[1] for t in net], 'r--', [t[0] for t in ill], [t[1] for t in ill], 'bs', [t[0] for t in ideo], [t[1] for t in ideo], 'g^')
plt.show()