import networkx as nx
import numpy as np
import copy
from random import choice
import matplotlib.pyplot as plt

# Corruption_Gcc    Crime_Gcc    PPI_gcc    RodeEU_gcc   BenchmarkNetwork
data = "data/RodeEU_gcc.txt"
E = np.loadtxt(data)
G = nx.MultiGraph()
G.add_edges_from(E)
# 得到图的邻接矩阵
A = nx.adjacency_matrix(G)
G_mat = A.todense()
vertex_number = len(G_mat)


# 得到每个点的度数
D = []
for i in range(0, len(G_mat)):
    each = G_mat[i][0]
    each[each > 0] = 1
    degree = each.sum()
    D.append(degree)
D = np.array(D)
# 得到点的全部标号
pre = list(np.loadtxt(data, dtype=np.int).flatten())
new_ = list(set(pre))
new_.sort(key=pre.index)


# 点标号的映射
def f(x):
    value = new_[x]
    return value


def karger_Min_Cut(graph, D):
    pair = []
    while np.sum(D > 0) > 2:
        # 随机选一个顶点
        u_beixuan = np.array(np.where(D > 0))[0]
        u = choice(u_beixuan)
        u_no = np.where(graph[u] > 0)[1]
        v = choice(u_no)    # 选出一条边
        pair.append((u,v))
        contract(graph, u, v, D)
    return D, pair


def contract(graph, u, v, D):
    # 更新点的度数
    D[u] = D[u] + D[v] - 2*graph[u, v]
    D[v] = 0
    # 删除uv相连的边（自环）
    graph[u, v] = 0
    graph[v, u] = 0
    v_ = np.where(graph[v] > 0)[1]  # 与v相连的顶点
    # 更新与v有边的点的列表
    for vertex in v_:
        if vertex != u and vertex != v:
            graph[vertex, u] = graph[vertex, u] + graph[vertex, v]
            graph[u, vertex] = graph[u, vertex] + graph[v, vertex]
            graph[vertex, v] = 0
            graph[v, vertex] = 0


# 规约找点：输入索引，返回索引
def find_nodes(array, u_list, v_list, nodes):
    v_ = []
    u_ = []
    for each in array:
        v_.append(v_list[each])
    nodes.extend(v_)
    for every in v_:
        u_.extend([i for i, x in enumerate(u_list) if x == every])
    if len(u_) == 0:     # 空数组,即已找到所有合并过的点
        return 0
    return u_

# 寻找点集
def find_set(set, u_list, v_list, nodes):
    result1 = find_nodes(set, u_list, v_list, nodes)
    while result1 != 0:
        result1 = find_nodes(result1, u_list, v_list, nodes)
    nodes = [f(x) for x in nodes]
    return nodes

# 绘图函数
def plot(matrix):
    mat = copy.deepcopy(matrix)
    arr = []
    for i in range(len(mat)):
        for j in range(len(mat)):
            if mat[i, j] > 0:
                mat[j, i] = 0
                number = mat[i, j]
                for m in range(number):
                    arr.append((f(i), f(j)))  # 默认0索引，现在加上
    # print(array)
    G = nx.MultiGraph(arr)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size= 550, node_color='r', alpha=1)
    ax = plt.gca()
    for e in G.edges:
        ax.annotate("",
                    xy=pos[e[0]], xycoords='data',
                    xytext=pos[e[1]], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="0.5",
                                    shrinkA=5, shrinkB=5,
                                    patchA=None, patchB=None,
                                    connectionstyle="arc3,rad=rrr".replace('rrr', str(0.05 * e[2])), ), )
    nx.draw_networkx_labels(G, pos, labels=None, font_size=20, font_color='k', font_family='sans-serif',
                            font_weight='normal', alpha=1.0, bbox=None, ax=None)
    # 保存为透明图像
    plt.savefig("图片.png", transparent=True)
    plt.show()


Matrix = []
pair_ = []
D_ = []
countMin = float('inf')
for i in range(150):
    if (i % 30 == 0):
        print('第', str(i), '局')
    MatCopy = copy.deepcopy(G_mat)   # 副本
    DCopy = copy.deepcopy(D)   # 副本
    D_no, pair_array = karger_Min_Cut(MatCopy, DCopy)    # x返回向量D和合并过的节点对
    count = np.max(D_no)
    if count < countMin:
        countMin = count
        Matrix.append(MatCopy)
        pair_.append(pair_array)
        D_.append(D_no)
print("karger_min_cut is " + str(countMin))

last = Matrix[-1]
degree = D_[-1]
node_set = ([i for i, x in enumerate(degree) if x != 0])
print(node_set)

last_pair = pair_[-1]
A = []
B = []
A.append(node_set[0])    # 点集1
B.append(node_set[1])    # 点集2
u_list = []
v_list = []
for m in range(len(last_pair)):
    each = last_pair[m]
    u = each[0]
    v = each[1]
    u_list.append(u)
    v_list.append(v)


# 输出点集
set1 = ([i for i, x in enumerate(u_list) if x == node_set[0]])
setA = find_set(set1, u_list, v_list, A)
print("点集1：", setA)

set2 = ([i for i, x in enumerate(u_list) if x == node_set[1]])
setB = find_set(set2, u_list, v_list, B)
print("点集2：", setB)

# 作图
plot(last)