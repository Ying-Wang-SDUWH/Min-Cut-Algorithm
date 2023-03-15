import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from random import choice
import copy


filename = "BenchmarkNetwork"
filename = "Corruption_Gcc"
# filename = "Crime_Gcc"
# filename = "PPI_gcc"
# filename = "RodeEU_gcc"

E = np.loadtxt("data/"+filename+".txt")
G = nx.Graph()
G.add_edges_from(E)
# 得到图的邻接矩阵
A = nx.adjacency_matrix(G)
G_mat = A.todense()
# 输出这个邻接矩阵
# print('邻接矩阵:\n',G_mat)
# 画出这张图
# nx.draw(G)


# 得到每个点的度数
D = []
for i in range(0, len(G_mat)):
    each = G_mat[i][0]
    each[each > 0] = 1
    degree = each.sum()
    D.append(degree)
D = np.array(D)

L = list(np.loadtxt("data/"+filename+".txt", dtype=np.int).flatten())
Array = list(set(L))
Array.sort(key=L.index)
# 点标号的映射
def f(x):
    value = Array[x]
    return value


def karger_Min_Cut(graph, D):
    pair = []
    while np.sum(D > 0) > 20:     # 设定Karger的阈值
        # 随机选一个顶点
        u_beixuan = np.array(np.where(D > 0))[0]
        u = choice(u_beixuan)
        u_no = np.where(graph[u] > 0)[1]
        v = choice(u_no)    # 选出一条边
        pair.append((u,v))
        contract_K(graph, u, v, D)
    return D, pair


def contract_K(graph, u, v, D):
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
    C = []
    D = []
    for each in array:
        C.append(v_list[each])
    nodes.extend(C)
    for every in C:
        D.extend([i for i, x in enumerate(u_list) if x == every])
    if len(D) == 0:     # 空数组
        return 0
    return D


# Stoer-Wagner算法
def GlobalMinCut(G_mat):
    # 记录每个结点
    D = np.arange(len(G_mat))
    # 初始化mincut为一个大数
    mincut = 10000
    E_ = []
    
    while len(G_mat) > 2:
        s,t,mc = MinCut(G_mat)
        
        if mc < mincut:
            mincut = mc
        
        # 当发现最小割为1时直接退出
        if mincut == 1:
            E_.append([D[t],D[t]])  # 相当于此时不要把点添加进去，同时保证E_中至少有一条边
            D = [D[s],D[t]]
            break
        
        E_.append([D[s],D[t]])         
        #合并s,t
        G_mat = contract_SW(G_mat,s,t)
        
        print(D[s],D[t],mincut)
        
        # 删除t点编号，把s点编号换到最后面
        if s<t:
            D = np.delete(D,t)
            zjl = D[-1]
            D[-1] = D[s]
            D[s] = zjl
        else:
            zjl = D[-1]
            D[-1] = D[s]
            D[s] = zjl
            D = np.delete(D,[t])
            
        # 输出剩余的点的标号
        print(D)
        
    return mincut,D,E_


# 求解任意st最小割的函数
def MinCut(G_mat):
    A = []
    a = np.random.randint(0,len(G_mat)-1)
    A.append(a)
    
    # 构造点集V-A记为V_no_
    V_no_ = list(np.arange(len(G_mat)))
    
    while(len(A)<len(G_mat)):
        
        V_no_.remove(A[-1])
        # print(V_no_)
        # print(A)
        
        A.append(V_no_[np.argmax(np.sum(G_mat[A][:,V_no_],axis=0))])

    s = A[-2]
    t = A[-1]
    mincut = np.sum(G_mat[A,t])-G_mat[A[-1],t]
    
    return s,t,mincut


def contract_SW(G_mat,s,t):
    new_mat = np.zeros((len(G_mat)+1,len(G_mat)+1))
    new_mat[:len(G_mat),:len(G_mat)] = G_mat
    for i in range(len(G_mat)):
        if i != s and i != t:
            new_mat[i,-1] = new_mat[i,s] + new_mat[i,t]
            new_mat[-1,i] = new_mat[s,i] + new_mat[t,i]
    new_mat = np.delete(new_mat, (s,t), axis = 0)
    new_mat = np.delete(new_mat, (s,t), axis = 1)
    return new_mat


Matrix = []
pair_ = []
D_ = []
# print(karger_Min_Cut(G_mat))
countMin = float('inf')
for i in range(100):
    if (i % 30 == 0):
        print('第', str(i), '局')
    MatCopy = copy.deepcopy(G_mat)   # 副本
    DCopy = copy.deepcopy(D)   # 副本
    count, pair_array = karger_Min_Cut(MatCopy, DCopy)    # 节点对
    count_number = np.max(count)
    if count_number < countMin:
        countMin = count_number
        Matrix.append(MatCopy)
        pair_.append(pair_array)
        D_.append(count)
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

# 点集1
set1 = ([i for i, x in enumerate(u_list) if x == node_set[0]])
result1 = find_nodes(set1, u_list, v_list, A)
while result1 != 0:
    result1 = find_nodes(result1, u_list, v_list, A)
A = [f(x) for x in A]    # 还原标号
print("点集1：", A)

# 点集2
set2 = ([i for i, x in enumerate(u_list) if x == node_set[1]])
result2 = find_nodes(set2, u_list, v_list, B)
while result2 != 0:
    result2 = find_nodes(result2, u_list, v_list, B)
B = [f(x) for x in B]    # 还原标号
print("点集2：", B)


# 作图
last = Matrix[-1]
arr = []
for i in range(len(last)):
    for j in range(len(last)):
        if last[i, j] > 0:
            last[j, i] = 0
            number = last[i, j]
            for m in range(number):
                arr.append([f(i), f(j)])
# print(arr)
'''
G=nx.MultiGraph(arr)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color = 'r', alpha = 1)
ax = plt.gca()
for e in G.edges:
    ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="-", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])),),)
nx.draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
plt.show()
'''


# 结点数降至阈值以下改用SW算法
G = nx.MultiGraph()
G.add_edges_from(arr)
# 得到图的邻接矩阵
A = nx.adjacency_matrix(G)
G_mat = A.todense()
mincut,D,E_ = GlobalMinCut(G_mat)

L = list(np.array(arr).flatten())
Array = list(set(L))
Array.sort(key=L.index)
# 点标号的映射
def F(x):
    value = Array[x]
    return value

# 映射到在原文件中的编号
D[0] = F(D[0])
D[1] = F(D[1])

array = []
for i in range(int(mincut)):
    array.append(D)

print("\nThe global minimum cut is:",D,mincut,"(在原文件中的标号)")


# 作图展示测试结果
G=nx.MultiGraph(array)
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color = 'r', alpha = 1)
ax = plt.gca()
for e in G.edges:
    ax.annotate("",
                xy=pos[e[0]], xycoords='data',
                xytext=pos[e[1]], textcoords='data',
                arrowprops=dict(arrowstyle="-", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle="arc3,rad=rrr".replace('rrr',str(0.3*e[2])
                                ),
                                ),
                )
nx.draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
# 保存为透明图像
plt.savefig("Advanced"+filename, transparent=True)
plt.show()


# 通过E_寻找两个超结点所含结点
G = nx.Graph()
G.add_edges_from(E_)
Set1 = set()
Set2 = set()
components=list(nx.connected_components(G))
for i in components[0]:
    Set1.add(F(i))

if len(components) > 1:
    for i in components[1]:
        Set2.add(F(i))

print(Set1,"\n",Set2)