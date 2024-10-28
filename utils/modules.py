import torch
from torch import nn
import numpy as np
import math


class Siren_Sin(torch.nn.Module):
    """
    Sine激活函数，激活时又omega修正频率，且第一层和非第一层有区别
    """

    def __init__(self, omega_0):
        self.omega_0 = omega_0
        super().__init__()

    def forward(self, x):
        x = self.omega_0 * x
        x = x.sin()
        return x


class NoAF(torch.nn.Module):
    """
    不使用激活函数
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_action(act_type):
    if act_type == "Sin":
        act = Siren_Sin(30.)
    elif act_type == "Tanh":
        act = nn.Tanh()
    elif act_type == "ReLU":
        act = nn.ReLU()
    elif act_type == "NoAF":
        act = NoAF()
    elif act_type == "Softsign":
        act = nn.Softsign()
    elif act_type == "Sigmoid":
        act = nn.Sigmoid()
    elif act_type == "LeakyReLU":
        act = nn.LeakyReLU()
    elif act_type == "GELU":
        act = nn.GELU()
    else:
        raise KeyError(f"Unknown activation function: {act_type}.")
    return act


class PositionalEncoding(nn.Module):
    """
    Torch.tensor: pos
    return pe_l * len(pos) * 2
    """
    def __init__(self, pe_b, pe_l, dim_coor = 2):
        super(PositionalEncoding, self).__init__()
        self.lbase, self.levels = pe_b, pe_l
        self.dim_pe = 2 * dim_coor * pe_l
    def forward(self, pos):
        pe_list = []
        for i in range(self.levels):  # [0,1,...,L-1]
            # b^i * index pi
            temp_value = pos * self.lbase ** (i) * math.pi  #
            pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
        return torch.stack(pe_list, 1)


class NetLayer(nn.Module):
    # 定义一个全链接层
    # in_features, out_features, bias, is_first = False, omega_0 = 30, act_type = "Sin", init_weight = None
    # 如果使用Sin作为激活函数，则需要注意初始化与激活时的计算, 如果是sin激活函数
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    def __init__(self, dim_in, dim_out, bias=True,
                 is_first=False, omega_0=30., act_type="Sin"):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = dim_in
        self.linear = nn.Linear(dim_in, dim_out, bias=bias)

        self.act = get_action(act_type)

        # Sin激活函数需要初始化网络
        if act_type == "Sin":
            print("INIT_WEIGHTS")
            self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, x):
        x = self.linear(x)
        return self.act(x)

# 这是一个PyTorch神经网络的类，它包括两个方法: init_weights和forward。init_weights方法用于初始化网络权重。"
# 如果这是第一层，则使用均匀分布从-1//dimin到1//dimin的范围来初始化权重。"
# 否则,使用均匀分布从-V6/(diminwo)到V6/(diminwo)的范围来初始化权重。"
# 其中,wo 是构造函数中传递的角频率参数，dimin 是输入维度。forward方法实现了前向传播过程。"
# 首先将输入数据 x传入线性层,然后应用激活函数并返回结果。这个类的作用是将输入数据嵌入映射到目标维度,并应用激活函数。"
# 在训练过程中,该类的实例将被多次调用以更新网络权重,以最小化损失函数。"

class Net(nn.Module):
    def __init__(self, dim_in, dim_hiddens, dim_out, first_omega_0=30., 
                 hidden_omega_0=30., act_type="Sin", out_activation=None):

        super().__init__()
        self.layers = []
        # 第一层
        self.layers.append(NetLayer(dim_in, dim_hiddens[0],
                                    is_first=True, omega_0=first_omega_0, act_type=act_type))

        for i in range(len(dim_hiddens)-1):
            # self.layers.append(nn.BatchNorm1d(dim_hiddens[i]))
            self.layers.append(NetLayer(dim_hiddens[i], dim_hiddens[i+1],
                                        is_first=False, omega_0=hidden_omega_0, act_type=act_type))
        # 最后一层
        self.layers.append(nn.BatchNorm1d(dim_hiddens[-1]))
        self.layers.append(NetLayer(dim_hiddens[-1], dim_out,
                                    is_first=False, omega_0=hidden_omega_0, act_type=out_activation))

        self.net = nn.Sequential(*self.layers)
# nn.Sequential 构造函数可以接受任意数量的子模块，并将它们按照顺序连接起来形成一个序列式神经网络。"
# 在这里，通过使用 *self.layers 将列表中的所有层解包为单独的参数，然后传递给 nn.Sequential。"
# 最终，self.net 成为一个包含所有层的 PyTorch 的 Sequential 对象，它将按照列表中的顺序依次应用于输入数据，并生成输出结果。"
        
    def forward(self, coords):
        output = self.net(coords)
        return output

class Net_multicls(nn.Module):
    def __init__(self, dim_in, dim_hiddens, dim_out, first_omega_0=30., 
                 hidden_omega_0=30., act_type="Sin", out_activation=None):

        super().__init__()
        self.layers = []
        self.dim_out = dim_out
        # 第一层
        self.layers.append(NetLayer(dim_in, dim_hiddens[0],
                                    is_first=True, omega_0=first_omega_0, act_type=act_type))

        for i in range(len(dim_hiddens)-1):
            self.layers.append(NetLayer(dim_hiddens[i], dim_hiddens[i+1],
                                        is_first=False, omega_0=hidden_omega_0, act_type=act_type))
        # 最后一层
        self.layers.append(NetLayer(dim_hiddens[-1], dim_out*2,
                                    is_first=False, omega_0=hidden_omega_0, act_type=out_activation))

        self.net = nn.Sequential(*self.layers)

        
    def forward(self, coords):
        output = self.net(coords)
        B = output.shape[0]
        output = output.reshape(B,2,-1)
        return output