import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler 
import numpy as np
from scipy.stats import truncnorm
from collections import defaultdict
from scipy.sparse import csr_matrix, diags
import matplotlib.pyplot as plt
import itertools
import os
import faiss
import time


def sigmoid(prob, xi):
    return 1/(1+np.exp(-xi * prob))



def load_and_preprocess_data_Hillstrom(data_path: str = r"data/Hillstrom/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"):
    """
    
    Args:
        
    Returns:
        pd.DataFrame: 预处理后的特征数据
    """
    X = pd.read_csv(data_path)
    
    # 删除不需要的列
    X = X.drop(columns=['segment', 'visit', 'conversion', 'spend'])
    
    NUMERIC_COLS = ['recency', 'history', 'mens', 'womens', 'newbie']
    CATEGORICAL_COLS = ['history_segment', 'zip_code', 'channel']
    feature_list = []
    for col in CATEGORICAL_COLS:
        enc = OneHotEncoder(drop='first')
        enc.fit(np.array(X[[col]]).reshape((-1, 1)))
        for k, name in enumerate(enc.get_feature_names_out([col])):
            X[name] = enc.transform(np.array(X[[col]]).reshape((-1, 1))).toarray()[:, k]
        feature_list.append(col)
    X.drop(feature_list, axis=1, inplace=True)
    # 保留数值列和新生成的one-hot列
    numeric_and_onehot = NUMERIC_COLS + [col for col in X.columns if col not in NUMERIC_COLS]
    X = X[numeric_and_onehot]
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X)
    X_t = X_t / 1.5
    X_t = pd.DataFrame(X_t)
    return X_t

def load_and_preprocess_data_Criteo(data_path: str = r"data/Criteo/criteo-uplift-v2.1.csv"):
    """
    Args:
        data_path: Criteo数据集路径
    Returns:
        pd.DataFrame: 预处理后的特征数据
    """
    X = pd.read_csv(data_path)
    
    
    X = X.drop(columns=['treatment', 'conversion', 'visit', 'exposure'])
    
    
    NUMERIC_COLS = [f'f{i}' for i in range(12)]
    X = X[NUMERIC_COLS]
    
    X = X.head(100000).reset_index(drop=True)
    
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X)
    X_t = np.tanh(X_t) * 2.5
    X_t = pd.DataFrame(X_t)
    
    return X_t

def load_and_preprocess_data(dataset_name='Hillstrom'):
    if dataset_name == 'Hillstrom':
        data = load_and_preprocess_data_Hillstrom()
    elif dataset_name == 'Criteo':
        data = load_and_preprocess_data_Criteo()
    else:
        data = load_and_preprocess_data_Hillstrom()
    return data




def neighbour_set(data, save_distance_matrix=False):
    
    
    delta=0.005
    
    data_np = np.ascontiguousarray(data.values.astype(np.float32))
    n_points = data_np.shape[0]
    dim = data_np.shape[1]
    
    print(f"开始计算邻居关系（FAISS），数据点数量: {n_points}")
    
    # 构建FAISS索引，尝试GPU加速
    index_flat = faiss.IndexFlatL2(dim)
    try:
        gpu_res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index_flat)
        gpu_index.add(data_np)
        print("使用FAISS GPU索引")
        search_index = gpu_index
    except (AttributeError, RuntimeError):
        index_flat.add(data_np)
        print("使用FAISS CPU索引")
        search_index = index_flat
    
    
    print("执行range_search...")
    t0 = time.time()
    lims, D, I = search_index.range_search(data_np, delta ** 2)
    print(f"range_search 完成，耗时: {time.time()-t0:.1f}秒")
    
    # 直接从FAISS输出构建CSR稀疏邻接矩阵
    print("构建稀疏邻接矩阵...")
    indptr = lims.astype(np.int64)
    indices = I.astype(np.int64)
    data_vals = np.ones(len(indices), dtype=np.float64)
    
    adj = csr_matrix((data_vals, indices, indptr),
                     shape=(n_points, n_points))
    adj.setdiag(0)
    adj.eliminate_zeros()
    
    
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    adj_norm = diags(1.0 / row_sums).dot(adj)
    
    n_with_neighbors = int(np.sum(row_sums > 1e-10))
    print(f"邻居计算完成，共找到 {n_with_neighbors} 个有邻居的节点")
    
    return adj_norm, None

def T_generate(data,xi,beta_nt,adj_norm=None):
    np.random.seed(42)
    
    beta_nt = beta_nt

    X = data   
    
    beta_for_T = np.random.normal(loc=-0.2,scale=0.01,size=X.shape[1])
    sigma_i = np.dot(X,beta_for_T).squeeze()
   
    if adj_norm is None:
        adj_norm , _ = neighbour_set(data)
    
    
    sigma_i_array = np.array(sigma_i).flatten()
    sigma_j = np.array(adj_norm.dot(sigma_i_array)).flatten()

    put_in_prob = (sigma_i + sigma_j * beta_nt) + 0.3
    final_prob = sigmoid(prob=put_in_prob,xi=xi)
    
    T = np.random.binomial(1, final_prob, X.shape[0])
    data_with_T = data.copy()
    data_with_T['T'] = T
    return T, data_with_T

def generate_inner_2(data):
    X = data.values
    d = X.shape[1]
    n = X.shape[0]
    
    # 使用numpy向量化操作生成二次项 - 加速版本
    # 创建所有特征组合的索引
    i_indices, j_indices = np.triu_indices(d)  # 获取上三角矩阵的索引
    
    # 使用广播一次性计算所有二次项
    X_i = X[:, i_indices]  # shape: (n, num_pairs)
    X_j = X[:, j_indices]  # shape: (n, num_pairs)
    inner_2 = X_i * X_j    # 元素级乘法，shape: (n, num_pairs)
    
    return inner_2

def generate_inner_3(data):
    X = data.values
    d = X.shape[1]
    n = X.shape[0]
    
    # 枚举所有三元组组合（带重复，保证对称性）
    combs = list(itertools.combinations_with_replacement(range(d), 3))
    num_triples = len(combs)
    
    # 初始化结果矩阵
    inner_3 = np.empty((n, num_triples))
    
    for idx, (i, j, k) in enumerate(combs):
        inner_3[:, idx] = X[:, i] * X[:, j] * X[:, k]
    
    return inner_3


def generate_inner_4(data):
    X = data.values
    d = X.shape[1]
    n = X.shape[0]
    
    # 枚举所有四元组组合（带重复）
    combs = list(itertools.combinations_with_replacement(range(d), 4))
    num_quads = len(combs)
    
    # 初始化结果矩阵
    inner_4 = np.empty((n, num_quads))
    
    for idx, (i, j, k, l) in enumerate(combs):
        inner_4[:, idx] = X[:, i] * X[:, j] * X[:, k] * X[:, l]
    
    return inner_4


def potential_outcome(data, m, beat_ny_0, beat_ny_1, select_omega, xi, beta_nt, adj_norm=None):
    np.random.seed(42)
    # 生成交互项数据
    inner_data_2 = generate_inner_2(data)
    inner_data_3 = generate_inner_3(data)
    inner_data_4 = generate_inner_4(data)

    beta_j_1 = np.random.binomial(1, 0.25, size=data.shape[1])
    beta_j_0 = np.random.binomial(1, 0.25, size=data.shape[1])
    beta_j_k_1 = np.random.binomial(1, 0.50, size=inner_data_2.shape[1])
    beta_j_k_0 = np.random.binomial(1, 0.40, size=inner_data_2.shape[1])
    beta_j_k_2_0 = np.random.binomial(1, 0.06, size=inner_data_3.shape[1])
    beta_j_k_2_1 = np.random.binomial(1, 0.09, size=inner_data_3.shape[1])
      
       
    #得到交互性的处理效应 
    gamma_0 = np.matmul(data,beta_j_1).squeeze() + np.matmul(inner_data_2,beta_j_k_0).squeeze() + np.matmul(inner_data_3,beta_j_k_2_0).squeeze()
    gamma_1 = np.matmul(data,beta_j_0).squeeze() + np.matmul(inner_data_2,beta_j_k_1).squeeze() + np.matmul(inner_data_3,beta_j_k_2_1).squeeze()
    df_raw_gamma_0 = pd.DataFrame(gamma_0,index=data.index)
    df_raw_gamma_1 = pd.DataFrame(gamma_1,index=data.index)
    
   
    # 得到T和含有T的数据为下一步邻居效应做准备
    T , data_with_T = T_generate(data=data,xi=xi,beta_nt=beta_nt,adj_norm=adj_norm)
    if adj_norm is None:
        adj_norm , _ = neighbour_set(data)

    T_value = data_with_T['T'].values
  
    # 稀疏矩阵乘法：一次性计算所有点的邻居gamma均值
    gamma_0_array = np.array(gamma_0).flatten()
    gamma_1_array = np.array(gamma_1).flatten()
    
    neighbor_mean_gamma_0 = np.array(adj_norm.dot(gamma_0_array)).flatten()
    neighbor_mean_gamma_1 = np.array(adj_norm.dot(gamma_1_array)).flatten()
    
    # 用mask选择：T==0的用gamma_0邻居均值，T==1的用gamma_1邻居均值
    add_bias_value_neighbor_0 = np.where(T_value == 0, neighbor_mean_gamma_0, 0)
    add_bias_value_neighbor_1 = np.where(T_value == 1, neighbor_mean_gamma_1, 0)

    
    gamma_0 = gamma_0 + beat_ny_0 * add_bias_value_neighbor_0
    gamma_1 = gamma_1 + beat_ny_1 * add_bias_value_neighbor_1
    
    # 生成扰动项
    epsilon_0 = np.random.normal(0, 0.1, data_with_T.shape[0])
    epsilon_1 = np.random.normal(0, 0.1, data_with_T.shape[0])
    
    # Y的生成
    y_0 = gamma_0 + epsilon_0
    y_1 = gamma_1 + epsilon_1
    cate = gamma_1 - gamma_0
    y = T * y_1 + (1-T) * y_0
    
    # 这里开始对X进行设计
    # 设计settingA
    # 向下取整
    m = int(np.floor(m * data.shape[1]))    
    drop_columns = np.random.choice(data.columns, size=m, replace=False)
    X_obs = data.drop(columns=drop_columns)

    #设计settingB
    X_obs = X_obs + np.random.normal(loc=0, scale=(select_omega/8), size=X_obs.shape[1]) 

    
    
    return X_obs, T, y, gamma_0, gamma_1, cate

    

    
    
    




if __name__ == "__main__":
    import sys
    
    # 模式选择：'all' 一口气跑12个组合，'single' 跑单个参数
    run_mode = 'all'  # 'all' or 'single'
    
    # 选择数据集
    dataset_name = 'Criteo'
    data = load_and_preprocess_data(dataset_name=dataset_name)
    
    # 计算邻居字典（只算一次，作为中间变量传递复用）
    adj_norm, _ = neighbour_set(data, save_distance_matrix=False)
    
    # 固定参数
    beta_nt = 0.2
    
    if run_mode == 'all':
        # 12个参数组合，一口气跑完
        param_configs = [
            # hidden (3): 变化m
            {'bias_type': 'hidden', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            {'bias_type': 'hidden', 'xi': 0, 'm': 0.3, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            {'bias_type': 'hidden', 'xi': 0, 'm': 0.5, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            # measurement (3): 变化select_omega
            {'bias_type': 'measurement', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            {'bias_type': 'measurement', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 2.4},
            {'bias_type': 'measurement', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 3.6},
            # selection (3): 变化xi
            {'bias_type': 'selection', 'xi': 0.8, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            {'bias_type': 'selection', 'xi': 1.6, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            {'bias_type': 'selection', 'xi': 2.4, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            # spillover (3): 变化beat_ny_0/beat_ny_1
            {'bias_type': 'spillover', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.4, 'beat_ny_1': 0.8, 'select_omega': 1.2},
            {'bias_type': 'spillover', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.5, 'beat_ny_1': 0.95, 'select_omega': 1.2},
            {'bias_type': 'spillover', 'xi': 0, 'm': 0.1, 'beat_ny_0': 0.6, 'beat_ny_1': 1.1, 'select_omega': 1.2},
        ]

        for idx, cfg in enumerate(param_configs):
            print(f"\n{'='*60}")
            print(f"[{idx+1}/12] bias_type={cfg['bias_type']}, xi={cfg['xi']}, m={cfg['m']}, "
                  f"beat_ny_0={cfg['beat_ny_0']}, beat_ny_1={cfg['beat_ny_1']}, select_omega={cfg['select_omega']}")
            print(f"{'='*60}")
            
            X_obs, T, y, gamma_0, gamma_1, cate = potential_outcome(
                data=data, m=cfg['m'], beat_ny_0=cfg['beat_ny_0'], beat_ny_1=cfg['beat_ny_1'],
                select_omega=cfg['select_omega'], xi=cfg['xi'], beta_nt=beta_nt, adj_norm=adj_norm)
            
            T_series = pd.Series(T, name='T', index=X_obs.index)
            y_series = pd.Series(y, name='y', index=X_obs.index)
            gamma_0_series = pd.Series(gamma_0, name='gamma_0', index=X_obs.index)
            gamma_1_series = pd.Series(gamma_1, name='gamma_1', index=X_obs.index)
            cate_series = pd.Series(cate, name='cate', index=X_obs.index)
            
            final_df = pd.concat([X_obs, T_series, y_series, gamma_0_series, gamma_1_series, cate_series], axis=1)
            
            output_dir = os.path.join('data', dataset_name, cfg['bias_type'])
            output_filename = (f"xi={cfg['xi']}_m={cfg['m']}_beat_ny_0={cfg['beat_ny_0']}"
                              f"_beat_ny_1={cfg['beat_ny_1']}_select_omega={cfg['select_omega']}_df.csv")
            output_path = os.path.join(output_dir, output_filename)
            
            os.makedirs(output_dir, exist_ok=True)
            final_df.to_csv(output_path, index=False)
            print(f"数据已保存到: {output_path}")
    
    else:
        # 单独模式：跑单个参数组合
        bias_type = 'hidden'  # selection, spillover, hidden, measurement
        xi = 0          # 参数选择 0 0.8 1.6 2.4 selection
        m = 0.1         # 参数选择 0.1, 0.3, 0.5 hidden
        beat_ny_0 = 0.4 # 参数选择 0.4 0.5 0.6 spillover
        beat_ny_1 = 0.8 # 参数选择 0.8 0.95 1.1 spillover
        select_omega = 1.2  # 参数选择 1.2 2.4 3.6 measurement
        
        X_obs, T, y, gamma_0, gamma_1, cate = potential_outcome(
            data=data, m=m, beat_ny_0=beat_ny_0, beat_ny_1=beat_ny_1,
            select_omega=select_omega, xi=xi, beta_nt=beta_nt, adj_norm=adj_norm)
        
        T_series = pd.Series(T, name='T', index=X_obs.index)
        y_series = pd.Series(y, name='y', index=X_obs.index)
        gamma_0_series = pd.Series(gamma_0, name='gamma_0', index=X_obs.index)
        gamma_1_series = pd.Series(gamma_1, name='gamma_1', index=X_obs.index)
        cate_series = pd.Series(cate, name='cate', index=X_obs.index)
        
        final_df = pd.concat([X_obs, T_series, y_series, gamma_0_series, gamma_1_series, cate_series], axis=1)
        
        output_dir = os.path.join('data', dataset_name, bias_type)
        output_filename = f'xi={xi}_m={m}_beat_ny_0={beat_ny_0}_beat_ny_1={beat_ny_1}_select_omega={select_omega}_df.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            print(f"文件已存在，使用现有数据: {output_path}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            final_df.to_csv(output_path, index=False)
            print(f"数据已保存到: {output_path}")
