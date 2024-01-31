
import time
import os
import numpy as np
import scipy.sparse as sp


#------------------------------file data--------------------------------------
def read_knowledge_data(file_dir, file_name):
    '''file data format:`entity_head relation entity_tail`\n
    or tag_assignment:`user item tag`
    '''
    start = time.time()
    file = os.path.join(file_dir, file_name)
    data = np.loadtxt(file, dtype=np.int32)
    uniq_data = np.unique(data, axis=0)

    print(f"got data from {file}, time spend:{(time.time()-start)/60:.2}")
    print(f"\tfile data info [repeat knowledge {len(data)-len(uniq_data)}]")
    return uniq_data


def read_interaction_data(file_dir, file_name):
    '''file data format:`u i1 i2 ...\n`'''
    start = time.time()
    u_items_dict = dict()
    rep_u, rep_i = 0, 0
    file = os.path.join(file_dir, file_name)

    with open(file, "r") as f:
        for line in f.readlines():
            data = [int(x) for x in line.strip().split(' ')]
            u, item = data[0], data[1:]
            items = list(set(item))
            if len(items) > 0:
                if u in u_items_dict.keys():
                    rep_u += 1
                    item = u_items_dict[u] + items
                    items = list(set(item))
                u_items_dict[u] = items

            rep_i += len(item) - len(items)

    print(f"got data from {file},time spend:{(time.time()-start)/60:.2}")
    print(f"\tfile data info [repeat user:{rep_u},repeat item:{rep_i}]")
    return u_items_dict


#----------------------------convert np dict sp----------------------------------------
def to_sparse_adj(row, col, shape):
    val = np.ones_like(row)  # 返回一个用1填充的跟输入形状和类型一致的数组
    adj = sp.coo_matrix((val, (row, col)), dtype=np.float32, shape=shape)  # 根据数据集生成交互矩阵，维度是shape，这里行数是user的个数，列数是item的个数
    return adj
# 其中的(val, (row, col))，是指在矩阵的(row, col)这个位置使用val中的1标记，也即只要这个位置的user和item有交互历史就标记1


def dict2np_array(data_dict):
    new_data = []
    for u, i in data_dict.items():
        ui = list(zip([u] * len(i), i))
        new_data.extend(ui)

    column_info(new_data)
    return np.array(new_data)


#------------------------------tgcn sample--------------------------------------
def neighbor_sample(matrix, k):
    '''
    对交互矩阵的`每一行`非0元素下标 `有放回地`采样K个,如果没有，则采样k个0\n
    return: [data,weight] 采样的邻域下标, 和对应的权重
    '''
    matrix = matrix.tocsr()  # 压缩矩阵
    data = np.zeros((matrix.shape[0], k), dtype=np.int)  # 返回返回一个用0填充的跟输入形状和类型一致的数组
    weight = np.zeros((matrix.shape[0], k), dtype=np.int)
    for i in range(matrix.shape[0]):
        nonzeroId = matrix[i].nonzero()[1]  # nonzero()函数从给定矩阵返回非零的索引值，返回的是两个数组，一个是行下标，一个是列下标，这里[1]表示我们只要矩阵第i行中不为0的列下标
        if len(nonzeroId):
            sampleId = np.random.choice(nonzeroId, k)  # 从nonzeroId中随机取k个元素(下标)
            # 将采样下标整体加一
            data[i] = sampleId + 1

            weight[i] = matrix[i].toarray()[0, sampleId]

    return [data, weight]


def all_neighbor_sample(X):
    matrix, max_deg = X
    matrix = matrix.tocsr()
    #max_deg = int(max(matrix.getnnz(1)))
    data = np.zeros((matrix.shape[0], max_deg), dtype=np.int)
    weight = np.zeros((matrix.shape[0], max_deg), dtype=np.int)
    for i in range(matrix.shape[0]):
        nonzeroId = matrix[i].nonzero()[1]
        x = len(nonzeroId)
        if x == 0:
            continue
        elif x < max_deg:
            sampleId = np.random.choice(nonzeroId, max_deg)
        else:
            sampleId = np.random.choice(nonzeroId, max_deg, replace=False)

        data[i] = sampleId + 1
        weight[i] = matrix[i].toarray()[0, sampleId]

    return [data, weight]

#----------------------------------------------------------------------------
def column_info(data_list):
    data = np.array(data_list)
    print(f"\t[{data.shape[0]}]:(unique,min,max)")  # data.shape 返回的是元组，data.shape[0]返回的是行数，data.shape[1]返回的是列数
    max_list = []
    for i in range(data.shape[1]):
        col = data[:, i]  # 第i列的所有数据
        max_list.append(max(col))
        print(f"\tcolumn {i}:[{len(set(col)),min(col),max(col)}]")  # 返回该列数据的详细信息，有多少行，最小值和最大值

    return max_list


def dict_info(dict_data):
    user, item = [], []
    for u, i in dict_data.items():  # items()以列表返回可遍历的(键,值)
        user.extend([u] * len(i))  # 同一个user会和很多item交互，如果一个user分别和10个item交互，则需要添加10行user，对应的item与之一一对应
        item.extend(list(i))  # item与user一一对应
    data = np.stack([user, item]).T  # 堆叠user和item这两个数组，使之变成一个二维数组，然后转置变成我们想要的user：item的形式
    max_list = column_info(data)

    return data, max_list


#--------------------------------------------------------------------------

