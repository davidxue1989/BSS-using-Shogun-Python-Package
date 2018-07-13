"""
Created on Fri Jul 13 11:40:07 2018

@author: ses516
"""

import numpy as np
from shogun import SOBI, RealFeatures
import pylab as plt

M = 4000         # mode resolution
N = 4000         # time increments

# Rank
K = 4

# Initiation of A and B (A*B=Y)
A_var = np.random.random((M,K))
B_var = np.random.random((N,K))

# additive noise
noise_lvl = 0.10

# mode shapes frequencies and initial phases
af = np.array([1,2,3,4])
bf = np.array([2,3,5,7])
df = np.array([4,3,2,1])
cf = np.array([0.5,1.24,2.78,.125])

# damping ratios 
dampf = np.array([0.01,0.01,0.01,0.01])
T = 50
t_A = np.linspace(0,T,M)      
t_B = np.linspace(0,T,N)     
A = np.zeros((M,K))
B = np.zeros((N,K))

for i in range(4):
    a = np.sin(af[i]*np.pi*t_A/T)
    b = df[i]*np.multiply(np.exp(-1*dampf[i]*t_B*bf[i]),np.sin(bf[i]*np.pi*2*t_B+cf[i]))
    A[:,i] = a      # Actual A matrix
    B[:,i] = b      # Actual B matrix

Y_org = np.matmul(A,np.transpose(B))    # Dense Matrix    (A * B)

##  add noise to observations
noise = np.random.normal(loc=0.0, scale=0.2, size=B.shape)
B = B + noise

# Mixing Matrix
A = np.random.normal(size=(K,K))

# Mix Signals
X = np.dot(A,B.T)
mixed_signals = RealFeatures(X)

# Separating
sobi = SOBI()
signals = sobi.apply(mixed_signals)
S_ = signals.get_feature_matrix()
A_ = sobi.get_mixing_matrix();

plt.figure('mix')
plt.plot(X.T); plt.show()

plt.figure('separated')
plt.plot(S_.T); plt.show()

dt = 0.0125
plt.figure('psd')
for i in range(4):
    plt.psd(S_[i,:].T, 1024, 1 / dt)
plt.xlim((0,10)); plt.show()
