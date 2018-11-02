import numpy, math
import matplotlib.pyplot as plt
import numpy as np

from sympy import *
init_printing()



###The function defined below includes the Inverse Transform Sampling to generate the samples from the conditional distribution.
###x and y are set as random variables from the uniform distribution. After x and y are calculated, they are stored in matrix m


def gibbs(N,thin, theta):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    m = np.zeros(shape= (N,2))
    for i in range(N):
        U = numpy.random.uniform()
        m[i] = ((-1/y)*math.log(1-U*(1-math.exp(-theta*y)))),(-1/x)*math.log(1-U*(1-math.exp(-theta*x)))
        y = (-1/x)*math.log(1-U*(1-math.exp(-theta*x)))
        x = (-1/y)*math.log(1-U*(1-math.exp(-theta*y)))
    return(m)


m = gibbs(500,10,5)
plt.hist([row[0]for row in m])
print(f"Expection=", numpy.mean([row[0] for row in m]))

m = gibbs(5000,10,5)
plt.hist([row[0] for row in m])
print(f"Expection=", numpy.mean([row[0] for row in m]))

m = gibbs(50000,10,5)
plt.hist([row[0] for row in m])
print(f"Expection=", numpy.mean([row[0] for row in m]))
