import torch
import numpy as np
from abc import ABC, abstractmethod


class Process(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def run(cls, label_array : np.ndarray,) -> np.ndarray:
        pass


def loss_distance(index_sub, label_index_sub, decimal_places=8):
    scaling_factor = 10 ** decimal_places

    # 將輸入數據轉換為 NumPy 數組
    XA = np.array(index_sub, dtype=np.int64)
    XB = np.array(label_index_sub, dtype=np.int64)

    # 在維度 1 上增加一個維度
    XA = XA[:, np.newaxis, :]
    # 在維度 0 上增加一個維度
    XB = XB[np.newaxis, :, :]

    # 廣播機制：將 XA 和 XB 的大小擴展為相同的形狀
    a_int = (XA * scaling_factor).astype(np.int64)
    b_int = (XB * scaling_factor).astype(np.int64)

    # 計算成對距離
    distances = np.linalg.norm(a_int - b_int, axis=-1)

    # 轉換為浮點數並按比例縮放
    distances_float = distances.astype(np.float32) / scaling_factor

    # 計算每一行的最小值並四捨五入
    loss_min = np.round(np.min(distances_float, axis=1), decimals=5)

    return loss_min


def data_translate(slice):
    slice = np.swapaxes(slice, 0, 1)
    # TMU scans need to be flipped
    slice = np.flip(slice, 0)
    slice = np.flip(slice, 1)
    return slice


def inverse_data_translate(slice):
    slice = np.swapaxes(slice, 1, 0)
    # TMU scans need to be flipped
    slice = np.flip(slice, 0)
    slice = np.flip(slice, 1)
    return slice


def left_right_translate(slice):
    index_3001 = np.argwhere(slice == 3001)
    index_4001 = np.argwhere(slice == 4001)
    if index_3001[:, 1].min() < index_4001[:, 1].min():
        return False, slice
    else:
        return True, np.flip(slice, 1)


def inverse_left_right_translate(flip, slice):
    if flip:
        return np.flip(slice, 1)
    else:
        return slice