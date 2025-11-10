#%%
import numpy as np
import matplotlib.pyplot as plt

N = 100000
sigma = 5
mu = 2

x_1 = np.random.rand(N)
x_2 = np.random.rand(N)

r = sigma * np.sqrt(-2 * np.log(x_1))
theta = 2 * np.pi * x_2

u = r * np.cos(theta)
v = r * np.sin(theta)

y_1 = mu + u
y_2 = mu + v

normal = lambda x: np.exp(-(x- mu) ** 2 / (2* sigma**2)) / (sigma * np.sqrt(2 * np.pi))
compare = np.linspace(-3, 3, N)

counts1, bins1 = np.histogram(y_1, bins=50)
deltax1 = bins1[1] - bins1[0]
pdf1 = (counts1 / N) / deltax1
bin_mid1 = (bins1[:-1] + bins1[1:]) / 2

counts2, bins2 = np.histogram(y_2, bins=50)
deltax2 = bins2[1] - bins2[0]
pdf2 = (counts2 / N) / deltax2
bin_mid2 = (bins2[:-1] + bins2[1:]) / 2

plt.hist(y_1, bins=50, alpha=0.5, label='cos', color='r')
plt.hist(y_2, bins=50, alpha=0.5, label='sin', color='g')
plt.legend()
plt.show()
plt.clf()

plt.plot(bin_mid1, pdf1, label='cos', color='r', alpha=0.5)
plt.plot(compare, normal(compare), label='normal', color='b')
plt.plot(bin_mid2, pdf2, label='sin', color='g', alpha=0.5)
plt.legend()
plt.show()
plt.clf()