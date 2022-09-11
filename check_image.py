from PIL import Image
import numpy as np


w, h = 512, 512
data = np.zeros((h, w, 3), dtype=np.uint8)
data[0:200, 0:100] = [255, 0, 0] # red patch in upper left





# A = np.load('Case1.npz')
# # B = np.load('final2.npz')
# # print(A.files)
# # print(A['velocity'].shape)
# # data = A['velocity']
# print(data.shape)
# abc = A['velocity'][:,50]
# xyz = A['velocity'][50,:]

# norm_abc = np.linalg.norm(abc)
# normal_array_abc = abc/norm_abc
# print(normal_array_abc)

# norm_xyz = np.linalg.norm(xyz)
# normal_array_xyz = xyz/norm_xyz
# print(normal_array_xyz)


# # print(abc, abc.shape)
# # print(xyz, xyz.sShape)
# for i in range(100):
#     data[i, :] = abc[i]

# for j in range(200):
#     data[:, j] = xyz[i]

# # data[0:100, 0:200] = [abc[:], xyz[:], 0]
# # data[0:100, :] = [A['velocity'][50,:], 0, 0]
print(data)

img = Image.fromarray(data, 'RGB')
# # img.save('my.png')
img.show()