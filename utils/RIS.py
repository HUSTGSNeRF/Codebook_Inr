import math
import numpy as np

'''
    三个问题：
        1. 入射方向距离是否要减去？现在没减，而且加减没区别对强度计算
        2. 出射方向的相位变化是否要抵消？现在抵消了，抵消与否略有区别
        3. 计算出射相位变化，俯仰角是否要随着cell变化，还是固定RIS板子的中心即可？现在是固定RIS中心
'''


def vector(p1, p2):
    return p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]


def vector_inner_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v2[2] * v1[2]


def vector_len(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def merge_phase(list_E, list_P):
    list_E = np.array(list_E)
    list_P = np.array(list_P)
    arrIndex = list_P.argsort()
    list_E = list_E[arrIndex]
    list_P = list_P[arrIndex]
    l = len(list_P)
    i = 0
    list_e = []
    list_p = []
    while i < l:
        e = list_E[i]
        p = list_P[i]
        i += 1
        while i < l and p == list_P[i]:
            e += list_E[i]
            i += 1
        list_e.append(e)
        list_p.append(p)

    return list_e, list_p


class feed_Ant:

    def __init__(self, A, q=1, f=5.8e9):
        """
        参数说明：
            A: 初始强度, 默认为1
            q: 天线的方向图形, 默认为1
            f: 中心频率, 默认为5.8e9
        """
        self.A = A
        print("feed Power{}".format(self.A))
        self.q = q
        c = 3e8  # 光速
        lambda_ = c / f * 1e3  # 波长
        self.k = 2 * math.pi / lambda_

    def phase(self, v):
        """
        参数说明：
            v_r: 一个方向向量, 由RIS中心指向某个单元i的方向
            v_u: 一个方向向量, 表示所计算电场强度的方向
        返回值：
            ph: 所偏移相位为e^(-j*ph)
        """
        ph = self.k * vector_len(v)
        return ph

    def strength(self, theta, v):
        # 角度均为弧度制，pi是180°
        E = self.A * (math.cos(theta)) ** self.q / vector_len(v)
        return E


class RIS_cell:

    def __init__(self, f=5.8e9):
        c = 3e8  # 光速
        lambda_ = c / f * 1e3  # 波长
        self.k = 2 * math.pi / lambda_
        self.tau = 1

    def phase(self, Phi_mn, v_r, v_u):
        # v_u = np.array(v_u) / vector_len(v_u)
        ph = Phi_mn - self.k * vector_inner_product(v_r, v_u)

        return ph

    def strength(self, theta_mn, theta):
        """
        参数说明：
            theta_mn: 入射cell的theta角
            theta: 出射的theta角
        """
        E = math.cos(theta) * math.cos(theta_mn) * self.tau

        return E


class RIS_env:

    def __init__(self, center_feed_ant, center_RIS, M=10, N=16, size=25, feed_ant_A=1):
        """
        参数说明：
            center_feed_ant: 馈源天线的中心, 一个坐标(x,y,z)
            center_RIS: 反射面的中心, 一个坐标(x,y,z)
            M, N: RIS板子的单元个数, 默认板子规格是M*N的
            size: 每个cell的大小, 默认25mm

        注: 所有坐标和距离都用单位mm来表示
        """
        self.p_feed = center_feed_ant
        self.p_RIS = center_RIS
        self.M = M
        self.N = N
        self.size = size
        print('size{} center_feed:{}'.format(str(size), center_feed_ant))
        self.feed_ant = feed_Ant(A=feed_ant_A)
        self.RIS_cell = RIS_cell()

        # 初始化所有cell中心的坐标（根据单元大小和中心坐标）
        # 这里默认RIS板子处于整个房间的z-y平面，因此，所有x轴坐标相同
        self.cell_point = self.cell_center_point(center_RIS, M, N, size)

    def unit_cell(self, p_cell, p_target, Phi_mn):
        """
        参数说明:
            p_feed: 馈源天线坐标
            p_cell: cell的中心坐标
            p_RIS: RIS的中心坐标
        """
        # 馈源天线到cell的过程
        v_feed2cell = vector(self.p_feed, p_cell)
        v_RIS2cell = vector(self.cell_point[self.M - 1, 0], p_cell)
        v_minux_x = (-1, 0, 0)

        ##### !!!!!!!!!!!!!!!!! 是否减去
        theta = math.acos(vector_inner_product(v_feed2cell, v_minux_x) / vector_len(v_feed2cell))
        feed_E = self.feed_ant.strength(theta, v_feed2cell)
        feed_Phase = self.feed_ant.phase(v_feed2cell)

        # cell调节之后的过程
        #### !!!!!!!!!!!!!!!!!!!!!!!!!!!RIS to target or cell to target
        v_cell2target = vector(self.p_RIS, p_target)
        v_x = (1, 0, 0)

        theta_target = math.acos(vector_inner_product(v_cell2target, v_x) / vector_len(v_cell2target))
        cell_E = self.RIS_cell.strength(theta, theta_target)
        cell_Phase = self.RIS_cell.phase(Phi_mn, v_RIS2cell, v_cell2target)

        # 最终的效果

        E = feed_E * cell_E / vector_len(v_feed2cell) ** 3 / vector_len(v_cell2target) ** 3
        Phase = feed_Phase + cell_Phase

        return E, Phase

    def whole_RIS(self, target):
        list_E = []
        list_Phase = []
        for i in range(self.M):
            for j in range(self.N):
                E, P = self.unit_cell(self.cell_point[i, j], target, self.matrix[i, j])
                list_E.append(E)
                list_Phase.append(P)
        list_E, list_Phase = merge_phase(list_E, list_Phase)

        return list_E, list_Phase

    def compute_EandP(self, target, Matrix):
        """
        参数说明：
            target: 计算强度和相位的目标点
            Matrix: M*N的矩阵, 表示RIS的状态, 0/pi
        """
        self.matrix = Matrix
        E, Phase = self.whole_RIS(target)
        a = 0
        b = 0

        for i in range(len(E)):
            a += E[i] * math.cos(Phase[i])
            b += E[i] * math.sin(Phase[i])
        return math.sqrt(a ** 2 + b ** 2)

    def cell_center_point(self, center_RIS, M, N, size):
        p_matrix = np.zeros((M, N, 3))

        if M % 2 == 0:
            for i in range(M):
                for j in range(N):
                    p_matrix[i, j, 0] = center_RIS[0]
                    p_matrix[i, j, 2] = center_RIS[2] + M / 2 * size - i * size - 1 / 2 * size
        else:
            for i in range(M):
                for j in range(N):
                    p_matrix[i, j, 0] = center_RIS[0]
                    p_matrix[i, j, 2] = center_RIS[2] + M // 2 * size - i * size

        if N % 2 == 0:
            for i in range(M):
                for j in range(N):
                    p_matrix[i, j, 1] = center_RIS[1] - N / 2 * size + j * size + 1 / 2 * size
        else:
            for i in range(M):
                for j in range(N):
                    p_matrix[i, j, 1] = center_RIS[1] - N // 2 * size + j * size

        return p_matrix

    def cell_p(self):
        return self.cell_point
