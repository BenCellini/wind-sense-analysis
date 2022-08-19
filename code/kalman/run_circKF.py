
import numpy as np
from kalman_filter import circKF
import matplotlib.pyplot as plt
import matplotlib
import utility

fs = 100
dt = 1 / fs

A = np.array([[1 , dt], [0 , 1]])
B = np.array([[(1/2)*(dt**2)], [dt]])
C = np.transpose(np.array([[1], [0]]))
D = np.array([0])
m, n = np.shape(C)

print('A : ')
print(A)
print('B : ')
print(B)
print('C : ')
print(C)

t = np.arange(0, 20, dt)
n_point = len(t)

freq = 0.2
amp = 0.5*np.pi
v = 1.5
x_true = -np.pi + amp*np.sin(2*np.pi*freq*t) + v*t
dx_true = 2*np.pi*freq*amp*np.cos(2*np.pi*freq*t) + v
d2x_true = -((2*np.pi*freq)**2)*amp*np.sin(2*np.pi*freq*t)

noise = np.random.normal(loc=0.0, scale=0.5*amp, size=n_point)
x_noise = x_true + noise

x_true_wrap = utility.wrapTo2Pi(x_true)
x_noise_wrap = utility.wrapTo2Pi(x_noise)

x_noise = x_noise_wrap
x_true = x_true_wrap

Q = 2 * np.eye(n, n)
print('Q:')
print(Q)

sigma_a = 50
R = sigma_a**2
print('R:')
print(R)

sigma_model = 0.9
P0 = (sigma_model**2) * np.eye(n, n)
print('P0:')
print(P0)

kf = circKF(A, B, C, D, Q, R)
X, K = kf.runFilter(x_noise, d2x_true, P0, x0=None, circ=True)

print('K:')
print(np.shape(K))

print('DONE')

fig, ax = plt.subplots(4, 1, figsize=(8, 7))
mksz = 1.5

ax[0].plot(t, x_noise, color='gray',
             linestyle='',
             linewidth=1,
             marker='.',
             markersize=mksz,
             alpha=0.5)

ax[0].plot(t, x_true, color='black',
             linestyle='',
             linewidth=1,
             marker='.',
             markersize=2*mksz,
             alpha=1)

ax[0].plot(t, X[:,0], color='blue',
             linestyle='',
             linewidth=1.5,
             marker='.',
             markersize=2*mksz,
             alpha=1)

ax[1].plot(t, dx_true, color='black',
             linestyle='',
             linewidth=1,
             marker='.',
             markersize=2*mksz,
             alpha=1)

ax[1].plot(t, X[:,1], color='blue',
             linestyle='',
             linewidth=1.5,
             marker='.',
             markersize=2*mksz,
             alpha=1)

ax[2].plot(t, d2x_true, color='red',
             linestyle='',
             linewidth=2,
             marker='.',
             markersize=2*mksz,
             alpha=1)

ax[3].plot(t, K[:, 0], color='lightblue',
             linestyle='-',
             linewidth=1.5,
             marker='',
             markersize=2*mksz,
             alpha=1)

ax[3].plot(t, K[:, 1], color='darkgreen',
             linestyle='-',
             linewidth=1.5,
             marker='',
             markersize=2*mksz,
             alpha=1)

plt.show()

