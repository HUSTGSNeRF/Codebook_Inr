import math
from utils.RIS import RIS_env
import copy
import numpy as np
from tqdm import tqdm
import os
import pandas as pd


def coordinate_conversion(height, angle, distance):
    """
    :param height: 以米为单位
    :param angle: 以角度为单位而不是以弧度为单位，右偏为正
    :param distance: 以米为单位
    :return: 以毫米为单位的笛卡尔坐标
    """ 
    # math.radians 度数转为弧度
    x = distance * math.cos(math.radians(angle))
    y = distance * math.sin(math.radians(angle))
    z = height
    return 1000 * x, 1000 * y, 1000 * z


## 求两向量夹角
def cal_angle(v1,v2):
    """求两向量的夹角
    Args:
        vec_a (np): 向量1
        vec_b (np): 向量2

    Returns:
        value: 角度值 0~360
    """
    cos_ = np.dot(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))
    sin_ = np.cross(v1, v2) / (np.linalg.norm(v1, 2) * np.linalg.norm(v2, 2))
 
    print(sin_,cos_)
    arctan2_ = np.arctan2(sin_, cos_)
    return arctan2_/np.pi



def get_tp_r(x,y,z): # 
    v0 = np.array([1,0,0])
    a = np.array([x,y,z])
    
    cos_ = np.dot(v0, a) / (np.linalg.norm(v0, 2) * np.linalg.norm(a, 2))
    angle_theta = np.arccos(cos_)
    
    v1 = np.array([1,0])
    b = np.array([y,z])
    cos_ = np.dot(v1, b) / (np.linalg.norm(v1, 2) * np.linalg.norm(b, 2))
    sin_ = np.cross(v1, b) / (np.linalg.norm(v1, 2) * np.linalg.norm(b, 2))
    angle_phi = np.arctan2(sin_, cos_)
    if angle_phi < 0:
        angle_phi += 2*np.pi
    return np.degrees(angle_theta), np.degrees(angle_phi)


def get_tp(x,y,z): # 
    """求向量与法线夹角和向量与y轴夹角
    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
    """
    #法线的夹角
    
    phi1 = np.array([x,y,z])
    phi2 = np.array([1,0,0])
    angle_phi = cal_angle(phi1,phi2)
    # 投影与y轴夹角
    the1 = np.array([y,z])
    the2 = np.array([1,0])
    angle_theta = cal_angle(the1,the2)
    
    return angle_phi, angle_theta



import numpy as np
import os

def codebook_coding(codebook):
    """
    将输入的码本进行编码，将变量空间从2^(M×N)降低到(M+N-1)
    输入数据的格式为<class 'numpy.ndarray'>
    先算列的操作，再算行的操作————这个很重要！！！
    翻转记1，不反转记-1
    """
    temp_data1 = codebook[:, 0]  # 现在这个是甜蜜的行！！！
    temp_data2 = codebook[-1, :]  # 现在这个是甜蜜的列！！！
    if temp_data1[-1] == 1:
        temp_data2 = (temp_data2 + 1) % 2  # 取反操作

    abs_operation = np.append(temp_data1, temp_data2)

    if abs_operation[0] == 0:
        rel_operation = abs_operation[1:]
    else:
        rel_operation = abs_operation[1:]
        rel_operation = (rel_operation + 1) % 2

    rel_operation = rel_operation.astype(int)

    return rel_operation


def codebook_decoding(code, MN=(10, 16)):
    """
    这是甜蜜的解编码
    """
    code[code == 0] = -1
    code = np.append([-1], code)

    initial_codebook = np.ones(MN) * -1

    temp_data1 = code[:MN[0]]
    temp_data2 = code[MN[0]:]

    # print(temp_data1)
    # print(temp_data2)

    mat_M = np.diag(temp_data1)
    mat_N = np.diag(temp_data2)

    de_codebook = np.matmul(mat_M, initial_codebook)
    de_codebook = np.matmul(de_codebook, mat_N)

    de_codebook = ((de_codebook + 1)/2).astype(int)

    return de_codebook


def encode_codebooks(codebook_lists,MN=(10, 16)):
    res = 0
    from tqdm import tqdm
    enc_codebook_lists = []
    for codebook in tqdm(codebook_lists):
        # print(codebook.shape)
        # codebook = codebook_lists[i]
        code = codebook_coding(codebook)
        # print(code)
        bbb = codebook_decoding(code, MN)
        no_equal = (bbb!=codebook).sum()
        if no_equal != 0:
            print(no_equal)
            # print(i)
            print(codebook)
            print(bbb)
            break
        enc_codebook_lists.append(code)
        res += no_equal
    print(res)
    enc_codebook_lists = np.array(enc_codebook_lists)
    enc_codebook_lists[enc_codebook_lists==-1] = 0
    return enc_codebook_lists


def decode_encodes(encodes_lists,MN=(10, 16)):
    # encodes_lists = np.load(encodes_root)
    from tqdm import tqdm
    denc_codebook_lists = []
    for encodes in tqdm(encodes_lists):
        codebook = codebook_decoding(encodes,MN)
        denc_codebook_lists.append(codebook)
    denc_codebook_lists = np.array(denc_codebook_lists)
    print(denc_codebook_lists.shape)
    return np.array(denc_codebook_lists)
    # np.save(encodes_root[:-4] + '_denc.npy', denc_codebook_lists)