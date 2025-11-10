import matplotlib.pyplot as plt
import math
import numpy as np

x=np.linspace(0,2*(np.pi),100)
'''to plot functions, we need something called as a "figure". A figure is bascically a window in which our plots are shown. Hence, we now learn how to create a figure'''
fig=plt.figure()
'''now we learn how to add subplots to the figure'''
ax1=fig.add_subplot(2,1,1)
ax2=fig.add_subplot(2,1,2)
ax1.plot(x,np.sin(x))
ax2.plot(x,np.cos(x))
'''to name the axes in each plot'''
ax1.set(xlabel="x",ylabel="sin(x)",title="sin(x)")
ax2.set(xlabel="x",ylabel="cos(x)",title="cos(x)")
plt.tight_layout()
plt.show()
