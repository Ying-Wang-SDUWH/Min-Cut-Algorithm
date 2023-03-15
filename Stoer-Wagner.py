import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Stoer-Wagner算法
def GlobalMinCut(G_mat):
    # 记录每个结点
    D = np.arange(len(G_mat))
    # 初始化mincut为一个大数
    mincut = 10000
    E_ = []
    
    while len(G_mat) > 2:
        s,t,mc = MinCut(G_mat)
        E_.append([D[s],D[t]])
        
        if mc < mincut:
            mincut = mc
        
        #合并s,t
        G_mat = contract(G_mat,s,t)
        
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
        # print(D)
        
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


def contract(G_mat,s,t):
    new_mat = np.zeros((len(G_mat)+1,len(G_mat)+1))
    new_mat[:len(G_mat),:len(G_mat)] = G_mat
    for i in range(len(G_mat)):
        if i != s and i != t:
            new_mat[i,-1] = new_mat[i,s] + new_mat[i,t]
            new_mat[-1,i] = new_mat[s,i] + new_mat[t,i]
    new_mat = np.delete(new_mat, (s,t), axis = 0)
    new_mat = np.delete(new_mat, (s,t), axis = 1)
    return new_mat


# filename = "BenchmarkNetwork"
# filename = "Corruption_Gcc"
# filename = "Crime_Gcc"
filename = "PPI_gcc"
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


print("The number of nodes is:",len(G_mat))
mincut,D,E_ = GlobalMinCut(G_mat)


L = list(np.loadtxt("data/"+filename+".txt", dtype=np.int).flatten())
Array = list(set(L))
Array.sort(key=L.index)
# 点标号的映射
def f(x):
    value = Array[x]
    return value

# 映射到原文件中的点
D[0] = f(D[0])
D[1] = f(D[1])
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
plt.savefig(filename, transparent=True)
plt.show()


# 通过E_寻找两个超结点所含结点
G = nx.Graph()
G.add_edges_from(E_)
Set1 = set()
Set2 = set()
components=list(nx.connected_components(G))
for i in components[0]:
    Set1.add(f(i))

if len(components) > 1:
    for i in components[1]:
        Set2.add(f(i))

print(Set1,"\n",Set2)