from matplotlib import pyplot as plt
import numpy as np


dataset = np.load('low1.npz')
labelset = np.load('high1.npz')
data = dataset['velocity']
data2 = labelset['velocity']


print(data.shape)
print(data2.shape)


fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(data, interpolation='nearest')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(data2, interpolation='nearest')


# plt.imshow(data, interpolation='nearest')
# plt.imshow(data2, interpolation='nearest')

plt.show()