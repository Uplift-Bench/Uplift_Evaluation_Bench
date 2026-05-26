import pandas as pd
from sklearn.preprocessing import OneHotEncoder , StandardScaler 
import numpy as np
import torch
from scipy.stats import truncnorm
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools
import os

# 旋钮1 xi
# 旋钮2 beat_nt
# 旋钮3 beat_ny
# 旋钮4 m
# 旋钮5 select_omega
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
    X_t = X_t / 3
    X_t = pd.DataFrame(X_t)
    return X_t

def load_and_preprocess_data_Bank(data_path: str = r"data/Bank/bank-additional-full.csv"):
    """
    Args:
        data_path: Data file path
    Returns:
        pd.DataFrame: Preprocessed feature data
    """
    X = pd.read_csv(data_path, sep=";")
    
    X = X.drop(columns=['y'])
    
    NUMERIC_COLS = ['age', 'duration', 'campaign', 'pdays', 'previous',
                    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    CATEGORICAL_COLS = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'day_of_week', 'poutcome']
    feature_list = []
    for col in CATEGORICAL_COLS:
        enc = OneHotEncoder(drop='first')
        enc.fit(np.array(X[[col]]).reshape((-1, 1)))
        for k, name in enumerate(enc.get_feature_names_out([col])):
            X[name] = enc.transform(np.array(X[[col]]).reshape((-1, 1))).toarray()[:, k]
        feature_list.append(col)
    X.drop(feature_list, axis=1, inplace=True)
    
    numeric_and_onehot = NUMERIC_COLS + [col for col in X.columns if col not in NUMERIC_COLS]
    X = X[numeric_and_onehot]
    scaler = StandardScaler()
    X_t = scaler.fit_transform(X)
    X_t = X_t / 3
    X_t = pd.DataFrame(X_t)
    
    return X_t
  
def load_and_preprocess_data(dataset_name='Hillstrom'):
    if dataset_name == 'Hillstrom':
        data = load_and_preprocess_data_Hillstrom()
    else:
        data = load_and_preprocess_data_Bank()
    return data




def neighbour_set(data,save_distance_matrix=False):
    """高度优化的邻居集合计算函数 - 使用张量操作减少循环"""
    
    delta=0.1

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 将数据转换为torch tensor并移到GPU
    data_tensor = torch.tensor(data.values, dtype=torch.float32, device=device)
    n_points = data_tensor.shape[0]
    
    print(f"开始高效计算邻居关系，数据点数量: {n_points}")
    
    # 使用defaultdict和set来避免重复邻居，提高效率
    neighbor_dict = defaultdict(set)
    distance_df = None
    
    # 根据数据大小和可用内存动态调整策略
    if n_points <= 8000:  # 小数据集：一次性计算
        print("使用一次性计算策略")
        # 计算所有点对之间的距离（一次性）
        # 使用broadcasting计算距离矩阵
        diff = data_tensor.unsqueeze(1) - data_tensor.unsqueeze(0)  # (n, n, features)
        distances = torch.norm(diff, dim=2)  # (n, n) - 这就是完整的距离矩阵
        
        # 如果需要保存距离矩阵
        if save_distance_matrix:
            print("保存完整距离矩阵...")
            distance_df = pd.DataFrame(distances.cpu().numpy())
        
        # 创建邻居mask：距离小于delta且不是自己
        neighbor_mask = (distances < delta) & (distances > 0)
        
        # 使用torch.nonzero一次性找到所有邻居对
        neighbor_pairs = torch.nonzero(neighbor_mask, as_tuple=False)
        
        # 高效构建邻居字典
        if len(neighbor_pairs) > 0:
            # 转换为numpy以便快速处理
            neighbor_pairs_np = neighbor_pairs.cpu().numpy()
            
            # 使用向量化操作快速添加邻居关系
            sources = neighbor_pairs_np[:, 0]
            targets = neighbor_pairs_np[:, 1]
            
            # 批量添加邻居关系
            for src, tgt in zip(sources, targets):
                neighbor_dict[src].add(tgt)
        
    else:  # 大数据集：分块计算但大幅优化
        print("使用分块计算策略")
        if save_distance_matrix:
            print("警告：大数据集保存距离矩阵会占用大量内存，建议设置save_distance_matrix=False")
            # 为大数据集创建距离矩阵（使用CPU以节省GPU内存）
            distance_matrix = torch.zeros((n_points, n_points), dtype=torch.float32)
        
        # 动态调整块大小基于可用内存
        max_memory_gb = 8  # 假设最大使用8GB内存
        features = data_tensor.shape[1]
        # 估算每个块的最大大小
        chunk_size = min(2000, int(np.sqrt(max_memory_gb * 1024**3 / (features * 4))))  # 4 bytes per float32
        
        # 使用更高效的分块策略
        for i in range(0, n_points, chunk_size):
            end_i = min(i + chunk_size, n_points)
            chunk_data_i = data_tensor[i:end_i]
             
            # 对于每个块，计算与所有后续点的距离（上三角）
            for j in range(i, n_points, chunk_size):
                end_j = min(j + chunk_size, n_points)
                chunk_data_j = data_tensor[j:end_j]
                
                # 使用broadcasting计算块内距离
                diff = chunk_data_i.unsqueeze(1) - chunk_data_j.unsqueeze(0)
                chunk_distances = torch.norm(diff, dim=2)
                
                # 如果需要保存距离矩阵，填充对应的块
                if save_distance_matrix:
                    # 保存上三角部分
                    distance_matrix[i:end_i, j:end_j] = chunk_distances.cpu()
                    # 由于距离矩阵是对称的，也填充下三角部分（除了对角线块）
                    if i != j:
                        distance_matrix[j:end_j, i:end_i] = chunk_distances.t().cpu()
                
                # 创建邻居mask
                neighbor_mask = chunk_distances < delta
                
                # 排除对角线（如果是同一个块）
                if i == j:
                    # 排除对角线和下三角
                    mask_size = min(end_i - i, end_j - j)
                    diag_mask = torch.eye(mask_size, device=device, dtype=torch.bool)
                    # 扩展到完整块大小
                    full_diag_mask = torch.zeros_like(neighbor_mask, dtype=torch.bool)
                    full_diag_mask[:mask_size, :mask_size] = diag_mask
                    
                    # 创建上三角mask
                    triu_mask = torch.triu(torch.ones_like(neighbor_mask, dtype=torch.bool), diagonal=1)
                    neighbor_mask = neighbor_mask & (~full_diag_mask) & triu_mask
                
                # 找到邻居对
                neighbor_pairs = torch.nonzero(neighbor_mask, as_tuple=False)
                
                # 批量添加邻居关系 - 使用set操作避免重复
                if len(neighbor_pairs) > 0:
                    pairs_np = neighbor_pairs.cpu().numpy()
                    
                    # 计算全局索引
                    global_i_list = pairs_np[:, 0] + i
                    global_j_list = pairs_np[:, 1] + j
                    
                    # 使用向量化操作批量添加双向邻居关系
                    for gi, gj in zip(global_i_list, global_j_list):
                        neighbor_dict[gi].add(gj)
                        neighbor_dict[gj].add(gi)
            
            # 清理GPU内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # 如果保存了距离矩阵，转换为DataFrame
        if save_distance_matrix:
            print("转换距离矩阵为DataFrame...")
            distance_df = pd.DataFrame(distance_matrix.numpy())
    
    # 转换set为list（保持原接口兼容性）
    neighbor_dict_final = {k: list(v) for k, v in neighbor_dict.items()}
    
    print(f"邻居计算完成，共找到 {len(neighbor_dict_final)} 个有邻居的节点")
    if save_distance_matrix and distance_df is not None:
        print(f"距离矩阵大小: {distance_df.shape}")
    
    return neighbor_dict_final, distance_df

def T_generate(data,xi,beta_nt):
    np.random.seed(42)
    
    beta_nt = beta_nt

    X = data   
    
    beta_for_T = np.random.normal(loc=-0.2,scale=0.01,size=X.shape[1])
    sigma_i = np.dot(X,beta_for_T).squeeze()
   
    neighbor_dict , _ = neighbour_set(data)
    # 使用numpy向量化计算邻居概率 - 加速版本
    sigma_j = np.zeros(len(data))
    sigma_i_array = np.array(sigma_i)
    
    for idx in data.index:  # 遍历每个数据点
        if idx in neighbor_dict and len(neighbor_dict[idx]) > 0: #判断idx是否在邻居字典中且有邻居
            sigma_idx = neighbor_dict[idx] #得到邻居索引
            # 使用numpy索引一次性获取所有邻居的sigma值并计算平均值
            neighbor_sigma_values = sigma_i_array[sigma_idx]
            sigma_j[idx] = np.mean(neighbor_sigma_values)
        # else: sigma_j[idx] 已经初始化为0
   

    put_in_prob = (sigma_i + sigma_j * beta_nt) + 0.3
    final_prob = sigmoid(prob=put_in_prob,xi=xi)
    
    T = np.random.binomial(1, final_prob, X.shape[0])
    data_with_T = data.copy()  # 创建副本而不是修改原始数据
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


def potential_outcome(data, m, beat_ny_0, beat_ny_1, select_omega, xi, beta_nt):
    np.random.seed(42)
    # 生成交互项数据
    inner_data_2 = generate_inner_2(data)
    inner_data_3 = generate_inner_3(data)

    beta_j_1 = np.random.binomial(1, 0.3, size=data.shape[1])
    beta_j_0 = np.random.binomial(1, 0.2, size=data.shape[1])
    beta_j_k_1 = np.random.binomial(1, 0.5, size=inner_data_2.shape[1])
    beta_j_k_0 = np.random.binomial(1, 0.2, size=inner_data_2.shape[1])
    beta_j_k_2_1 = np.random.binomial(1, 0.6, size=inner_data_3.shape[1])
    beta_j_k_2_0 = np.random.binomial(1, 0.2, size=inner_data_3.shape[1])
      
       
    #得到交互性的处理效应 
    gamma_0 = np.matmul(data,beta_j_1).squeeze() + np.matmul(inner_data_2,beta_j_k_0).squeeze() 
    gamma_1 = np.matmul(data,beta_j_0).squeeze() + np.matmul(inner_data_2,beta_j_k_1).squeeze() + np.matmul(inner_data_3,beta_j_k_2_1).squeeze()
    df_raw_gamma_0 = pd.DataFrame(gamma_0,index=data.index)
    df_raw_gamma_1 = pd.DataFrame(gamma_1,index=data.index)
    
   
    # 得到T和含有T的数据为下一步邻居效应做准备
    T , data_with_T = T_generate(data=data,xi=xi,beta_nt=beta_nt)
    neighbor_dict , _ = neighbour_set(data)

    T_df = pd.DataFrame(data_with_T['T'],index=data.index)
    T_value = T_df.values
  
    add_bias_value_neighbor_0 = np.zeros(len(data))
    add_bias_value_neighbor_1 = np.zeros(len(data))
    gamma_0_array = np.array(gamma_0)
    gamma_1_array = np.array(gamma_1)
    
    
    for idx in data.index:  # 遍历每个数据点
        if idx in neighbor_dict and len(neighbor_dict[idx]) > 0: #判断idx是否在邻居字典中且有邻居
            if T_value[idx] == 0:
                gamma_idx_0 = neighbor_dict[idx]#得到邻居索引，仿照上面生成T的代码
                gamma_value_0 = gamma_0_array[gamma_idx_0]
                add_bias_value_neighbor_0[idx] = np.mean(gamma_value_0)
            elif T_value[idx] == 1:
                gamma_idx_1 = neighbor_dict[idx]
                gamma_value_1 = gamma_1_array[gamma_idx_1]
                add_bias_value_neighbor_1[idx] = np.mean(gamma_value_1)

    
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
    # 参数设置
    # bias类型选择
    bias_type = 'spillover' # selection, spillover, hidden, measurement
    # selection bias 旋钮
    xi = 0 #参数选择 0 0.8 1.6 2.4 selection
    
    beta_nt = 0.2  #这个参数是固定的

    # hidden bias 旋钮
    m = 0.1  #参数选择 0.1,0.3,0.5 hidden
    # spillover bias 旋钮
    beat_ny_0 = 0.5  #参数选择  0.4 0.5 0.6  spillover
    beat_ny_1 = 0.95  #参数选择  0.8 0.95 1.1 spillover
    # measurement bias 旋钮
    select_omega = 1.2 #参数选择 1.2 2.4 3.6 measurement
    
    # 选择数据集
    # 用'Hillstrom'和'Bank'进行数据集切换，默认使用Hillstrom
    dataset_name = 'Bank'
    data = load_and_preprocess_data(dataset_name=dataset_name)
       
    neighbor_dict, _ = neighbour_set(data, save_distance_matrix=False)    
    X_obs, T, y, gamma_0, gamma_1, cate = potential_outcome(
        data=data, m=m, beat_ny_0=beat_ny_0, beat_ny_1=beat_ny_1,select_omega=select_omega, xi=xi, beta_nt=beta_nt)
    
    # 将numpy数组转换为pandas Series，并指定列名
    T_series = pd.Series(T, name='T', index=X_obs.index)
    y_series = pd.Series(y, name='y', index=X_obs.index)
    gamma_0_series = pd.Series(gamma_0, name='gamma_0', index=X_obs.index)
    gamma_1_series = pd.Series(gamma_1, name='gamma_1', index=X_obs.index)
    cate_series = pd.Series(cate, name='cate', index=X_obs.index)
    
    # 合并所有数据
    final_df = pd.concat([X_obs, T_series, y_series, gamma_0_series, gamma_1_series, cate_series], axis=1)
    
    # 构建相对路径保存结果
    output_dir = os.path.join('data', dataset_name, bias_type)
    output_filename = f'xi={xi}_m={m}_beat_ny_0={beat_ny_0}_beat_ny_1={beat_ny_1}_select_omega={select_omega}_df.csv'
    output_path = os.path.join(output_dir, output_filename)
    
    # 检查文件是否已存在
    if os.path.exists(output_path):
        print(f"文件已存在，使用现有数据: {output_path}")
    else:
        # 如果目录不存在则创建
        os.makedirs(output_dir, exist_ok=True)
        # 保存文件
        final_df.to_csv(output_path, index=False)
        print(f"数据已保存到: {output_path}")

    

   
  
   

    
    
    
    
    
    
    
