from os import walk
import numpy as np

path = '../mobilenetv2_nchw_origin/weights/'
filenames = next(walk(path), (None, None, []))[2] 
for file in filenames:
    data = np.load(path+file)
    prev_shape = data.shape
    data = data.flatten()
    if len(prev_shape) == 4:
        new_shape = (prev_shape[3], prev_shape[1], prev_shape[2], prev_shape[0])
        data = data.reshape(new_shape)
        np.save(path+file, data)
        print(path, file, prev_shape, new_shape)
    else:
        np.save(path+file, data)
        print(path, file, prev_shape)
