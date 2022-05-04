import numpy as np
import copy
import math

def create_matrix_flex(pixel, slice, dicts):
    n_pixel = pixel
    n_slice = slice
    size_x = dicts['size_x']
    size_y = dicts['size_y']
    size_z = dicts['size_z']
    stride_x = dicts['stride_x']
    stride_y = dicts['stride_y']
    stride_z = dicts['stride_z']
    times_x = size_x/stride_x
    times_y = size_y/stride_y
    times_z = size_z/stride_z
    total = np.zeros((n_pixel, n_pixel, 1), dtype=float)
    for layer_num in range(n_slice):
        layer = np.zeros((1, n_pixel), dtype=float)
        flag = 0
        for i in range(int(times_x)):
            temp = np.zeros((1, n_pixel), dtype=float)
            temp[:, :] = 1.0
            temp[:, size_x:n_pixel-size_x] = 1/((i+1)*times_x)
            for j in range(int(times_x)):
                temp[0][stride_x*j:stride_x*(j+1)] *= 1/float((j+1)*(i+1))
                k = -1 - j
                if j == 0:
                    temp[0][stride_x * k:] *= 1 / float((j + 1) * (i + 1))
                else:
                    temp[0][stride_x*k:stride_x*(k+1)] *= 1/float((j+1)*(i+1))
            if flag == 0:
                layer = temp
                flag = 1
            else:
                layer = np.concatenate((layer, temp), axis=0)
            for m in range(stride_y-1):
                layer = np.concatenate((layer, temp), axis=0)
        # print(layer.shape)
        middle = copy.deepcopy(temp)
        for k in range(n_pixel-size_y*2-1):
            temp = np.concatenate((middle, temp), axis=0)
        tail = layer[::-1]
        layers = np.concatenate((layer, temp, tail), axis=0)
        # print(layers.shape)
        if layer_num < size_z-stride_z:
            layers *= (1.0/(layer_num//stride_z+1))
        elif layer_num > n_slice-size_z+stride_z-1:
            layers *= (1.0/(math.ceil((n_slice-layer_num)/stride_z)))
        else:
            layers *= (1 / times_z)

        layers = layers[:, :, np.newaxis]
        if layer_num == 0:
            total = layers
        else:
            total = np.concatenate((total, layers), axis=-1)
    return total


def create_matrix(pixel, slice, target, step):
    n_pixel = pixel
    n_slice = slice
    size = target
    stride = step
    times = size/stride
    total = np.zeros((n_pixel, n_pixel, 1), dtype=float)
    for layer_num in range(n_slice):
        layer = np.zeros((1, n_pixel), dtype=float)
        flag = 0
        for i in range(int(times)):
            temp = np.zeros((1, n_pixel), dtype=float)
            temp[:, :] = 1.0
            temp[:, size:n_pixel-size] = 1/((i+1)*times)
            for j in range(int(times)):
                temp[0][stride*j:stride*(j+1)] *= 1/float((j+1)*(i+1))
                k = -1 - j
                if j == 0:
                    temp[0][stride * k:] *= 1 / float((j + 1) * (i + 1))
                else:
                    temp[0][stride*k:stride*(k+1)] *= 1/float((j+1)*(i+1))
            if flag == 0:
                layer = temp
                flag = 1
            else:
                layer = np.concatenate((layer, temp), axis=0)
            for m in range(stride-1):
                layer = np.concatenate((layer, temp), axis=0)

        middle = copy.deepcopy(temp)
        for k in range(n_pixel-size*2-1):
            temp = np.concatenate((middle, temp), axis=0)

        tail = layer[::-1]
        layers = np.concatenate((layer, temp, tail), axis=0)
        if layer_num < size-stride:
            layers *= (1.0/(layer_num//stride+1))
        elif layer_num > n_slice-size+stride-1:
            layers *= (1.0/(math.ceil((n_slice-layer_num)/stride)))
        else:
            layers *= (1 / times)

        layers = layers[:, :, np.newaxis]
        if layer_num == 0:
            total = layers
        else:
            total = np.concatenate((total, layers), axis=-1)
    return total


def create_matrix_ori():
    stride = 8
    total = np.zeros((512, 512, 1), dtype=float)
    for layer_num in range(64):
        layer = np.zeros((1, 512), dtype=float)
        flag = 0
        for i in range(4):
            temp = np.zeros((1, 512), dtype=float)
            temp[:, :] = 1.0
            temp[:, 32:480] = 1/((i+1)*4)
            for j in range(4):
                temp[0][stride*j:stride*(j+1)] *= 1/float((j+1)*(i+1))
                k = -1 - j
                if j == 0:
                    temp[0][stride * k:] *= 1 / float((j + 1) * (i + 1))
                else:
                    temp[0][stride*k:stride*(k+1)] *= 1/float((j+1)*(i+1))
            if flag == 0:
                layer = temp
                flag = 1
            else:
                layer = np.concatenate((layer, temp), axis=0)
            for m in range(7):
                layer = np.concatenate((layer, temp), axis=0)
        middle = copy.deepcopy(temp)
        for k in range(447):
            temp = np.concatenate((middle, temp), axis=0)
        tail = layer[::-1]
        layers = np.concatenate((layer, temp, tail), axis=0)
        if layer_num < 24:
            layers *= (1.0/(layer_num//8+1))
        elif layer_num > 40-1:
            layers *= (1.0/(math.ceil((64-layer_num)/8)))
        else:
            layers *= (1 / 4.0)

        layers = layers[:, :, np.newaxis]
        if layer_num == 0:
            total = layers
        else:
            total = np.concatenate((total, layers), axis=-1)
    return total


if __name__ == '__main__':
    create_matrix()
