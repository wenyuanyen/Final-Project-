{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy, math\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sympy import *\n",
    "init_printing()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "###The function defined below includes the Inverse Transform Sampling to generate the samples from the conditional distribution.\n",
    "###x and y are set as random variables from the uniform distribution. After x and y are calculated, they are stored in matrix m \n",
    "\n",
    "\n",
    "def gibbs(N,thin, theta):\n",
    "    x = np.random.uniform(0,1)\n",
    "    y = np.random.uniform(0,1)\n",
    "    m = np.zeros(shape= (N,2))\n",
    "    for i in range(N):\n",
    "        U = numpy.random.uniform()\n",
    "        m[i] = ((-1/y)*math.log(1-U*(1-math.exp(-theta*y)))),(-1/x)*math.log(1-U*(1-math.exp(-theta*x)))\n",
    "        y = (-1/x)*math.log(1-U*(1-math.exp(-theta*x)))\n",
    "        x = (-1/y)*math.log(1-U*(1-math.exp(-theta*y)))\n",
    "    return(m)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'np' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-14-63d765a661c4>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0mm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgibbs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m500\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m5\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhist\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mrow\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;32mfor\u001b[0m \u001b[0mrow\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mm\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"Expection=\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnumpy\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mrow\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mrow\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mm\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m<ipython-input-12-96fa5db858d7>\u001b[0m in \u001b[0;36mgibbs\u001b[0;34m(N, thin, theta)\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mgibbs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mN\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0mthin\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtheta\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m     \u001b[0mx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandom\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0muniform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m     \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrandom\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0muniform\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      8\u001b[0m     \u001b[0mm\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mzeros\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m=\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0mN\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mNameError\u001b[0m: name 'np' is not defined"
     ]
    }
   ],
   "source": [
    "m = gibbs(500,10,5)\n",
    "plt.hist([row[0]for row in m])\n",
    "print(f\"Expection=\", numpy.mean([row[0] for row in m]))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
