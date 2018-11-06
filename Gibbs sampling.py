### Name of Members: Weidi Pan, Wen Yuan Yen, Yuhei Koshino ####


import numpy, math
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable

from sympy import *
init_printing()


###The function defined below includes the Inverse Transform Sampling to generate the samples from the conditional distribution.
###x and y are set as random variables from the uniform distribution. After x and y are calculated, they are stored in matrix m


def gibbs(N, thin, B):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    m = np.zeros(shape= (N,2)) 
    for i in range(N):
        for j in range(thin):
            U = np.random.uniform()
            x = (-1/y)*math.log(1-U*(1-math.exp(-B*y)))
            U = np.random.uniform()
            y = (-1/x)*math.log(1-U*(1-math.exp(-B*x)) )                      
        m[i,:] = [x,y]
    return(m)


m2 = gibbs(500,10,5)
plt.hist(m2[:,0])
print(f"Expection=", np.mean(m2[:,0]))

m3 = gibbs(5000,10,5)
plt.hist(m3[:,0])
print(f"Expection=", np.mean(m3[:,0]))

m4 = gibbs(50000,10,5)
plt.hist(m4[:,0])
print(f"Expection=", np.mean(m4[:,0]))

table = PrettyTable()
table.title = "Gibb sampling estimate of E(X), B = 5"
table.add_column('T', [500,5000,50000])
table.add_column('Estimate', [np.mean(m2[:,0]), np.mean(m3[:,0]), np.mean(m4[:,0])])
print(table)
