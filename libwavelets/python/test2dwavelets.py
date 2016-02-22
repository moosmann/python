import numpy as np
import libwaveletspy as lw
import matplotlib.pyplot as plt

x0 = 10
y0 = 10
nx = 100
ny = 100
x = np.linspace(-x0, x0, nx, dtype=np.float64)
y = np.linspace(-y0, y0, ny, dtype=np.float64)
[xx, yy] = np.meshgrid(x, y)
xx = np.sqrt(xx ** 2 + yy ** 2)

data = np.sin(3 * xx) + np.exp(xx / 5) + np.cos(1 / (0.01 + xx))

dataCpy = data.flatten()
wavelets = np.zeros_like(dataCpy)
result = np.zeros_like(dataCpy)

print(data.shape, dataCpy.shape, wavelets.shape, result.shape)

lw.wavelet_transform2D(dataCpy.ctypes.data, nx, ny, wavelets.ctypes.data)
lw.invwavelet_transform2D(wavelets.ctypes.data, nx, ny, result.ctypes.data)

dataCpy = dataCpy.reshape((nx, ny))
wavelets = wavelets.reshape((nx, ny))
result = result.reshape((nx, ny))

plt.close("all")

if 1:
    xp = nx / 2
    plt.figure(1)

    plt.subplot(3, 1, 1)
    plt.plot(data[xp, :], label="data")
    plt.plot(result[xp, :], label="result")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(result[xp, :] - data[xp, :], label="result-data")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(wavelets[xp, :], label="wavelets")
    plt.legend()

if 1:
    plt.figure(2)
    plt.subplot(2, 2, 1)
    plt.imshow(data)
    plt.title("data")

    plt.subplot(2, 2, 2)
    plt.imshow(dataCpy)
    plt.title("dataCpy")

    plt.subplot(2, 2, 3)
    plt.imshow(wavelets)
    plt.title("wavelets")

    plt.subplot(2, 2, 4)
    plt.imshow(result)
    plt.title("result")

plt.show()
