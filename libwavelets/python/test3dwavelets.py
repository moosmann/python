import numpy as np
import libwaveletspy as lw
import matplotlib.pyplot as plt

# linear range and sample points
x0 = 10
y0 = 10
z0 = 10
nx = 100
ny = 100
nz = 100

# create grid
x = np.linspace(-x0, x0, nx, dtype=np.float64)
y = np.linspace(-y0, y0, ny, dtype=np.float64)
z = np.linspace(-z0, z0, ny, dtype=np.float64)
[xx, yy, zz] = np.meshgrid(x, y, z)
xx = np.sqrt(xx ** 2 + yy ** 2 + zz ** 2)

# create 3D data
data = np.sin(3 * xx) + np.exp(xx / 5) + np.cos(1 / (0.01 + xx))

# create copy of data since data array will be modified by calling wavelet_transform reshape it to 1D vector
dataCpy = data.flatten()
wavelets = np.zeros_like(dataCpy)
result = np.zeros_like(dataCpy)

print(data.shape, dataCpy.shape, wavelets.shape, result.shape)

# forward and inverse wavelet transform
lw.wavelet_transform3D(dataCpy.ctypes.data, nx, ny, nz, wavelets.ctypes.data)
lw.invwavelet_transform3D(wavelets.ctypes.data, nx, ny, nz, result.ctypes.data)

# reshape 1D vectors to 3D arrays
dataCpy = dataCpy.reshape((nx, ny, nz))
wavelets = wavelets.reshape((nx, ny, nz))
result = result.reshape((nx, ny, nz))

# print line plots
if 1:
    xp = nx / 2
    yp = ny / 2

    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(data[xp, yp, :], label="data")
    plt.plot(result[xp, yp, :], label="result")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(result[xp, yp, :] - data[xp, yp, :], label="result-data")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(wavelets[xp, yp, :], label="wavelets")
    plt.legend()

# print 2D cuts
if 1:
    xp = nx / 2
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.imshow(data[xp, :, :])
    plt.title("data")

    plt.subplot(2, 2, 2)
    plt.imshow(dataCpy[xp, :, :])
    plt.title("dataCpy")

    plt.subplot(2, 2, 3)
    plt.imshow(wavelets[xp, :, :])
    plt.title("wavelets")

    plt.subplot(2, 2, 4)
    plt.imshow(result[xp, :, :])
    plt.title("result")

plt.show()
