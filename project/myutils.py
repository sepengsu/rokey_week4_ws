def rad2deg(rad):
    deg = rad * 180 / np.pi
    deg = deg % 360
    if deg < 0:
        deg += 360
    return deg
import numpy as np
BOX_LIST = ['BOOTSEL','USB','CHIPSET','OSCILLATOR','HOLE1','HOLE2','HOLE3','HOLE4']
def dict_to_array(box_dict):
    box_list = []
    for name in BOX_LIST:
        box_list.append(box_dict[name])
    return np.array(box_list)