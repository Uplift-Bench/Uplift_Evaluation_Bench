from metric import *
import pandas as pd
import os
import numpy as np
import re
from sklearn.model_selection import train_test_split

def get_raw_data_path(folder_path=r"data", dataset_name='Bank', bias_type="selection", all_data=False):
    """
    读取指定文件夹下的所有CSV文件路径，支持all_data模式自动遍历所有bias和程度
    
    Args:
        folder_path (str): 数据文件夹路径
        dataset_name (str): 数据集名称 (Hillstrom 或 Bank)
        bias_type (str): 偏差类型，用于筛选特定的子文件夹
        all_data (bool): 是否读取所有数据，如果为True则忽略bias_type参数
        
    Returns:
        dict: 包含所有CSV文件路径的字典，按bias类型和程度分组
    """
    # 拼接完整路径: folder_path + dataset_name
    full_path = os.path.join(folder_path, dataset_name)
    
    file_paths_dict = {}
    
    # 检查目录是否存在
    if not os.path.exists(full_path):
        print(f"错误: 目录 '{full_path}' 不存在")
        return file_paths_dict
    
    # 检查是否为目录
    if not os.path.isdir(full_path):
        print(f"错误: '{full_path}' 不是一个目录")
        return file_paths_dict
    
    if all_data:
        # 遍历所有 bias 文件夹
        for bias_type in os.listdir(full_path):
            bias_path = os.path.join(full_path, bias_type)
            if not os.path.isdir(bias_path):
                continue
            # degree 分组
            degree_dict = {}
            for file in os.listdir(bias_path):
                if file.endswith('.csv'):
                    # 匹配"程度"部分（去掉最后的 _数字.csv）
                    match = re.match(r'(.+)_\d+\.csv$', file)
                    if match:
                        degree_prefix = match.group(1)
                    else:
                        degree_prefix = file.split('.csv')[0]
                    if degree_prefix not in degree_dict:
                        degree_dict[degree_prefix] = []
                    degree_dict[degree_prefix].append(os.path.join(bias_path, file))
            # 存入总 dict
            for degree, files in degree_dict.items():
                key = f"{bias_type}_{degree}"
                file_paths_dict[key] = files
                print(f"找到: {key} -> {len(files)} 个文件")
    else:
        # 只读取指定bias_type的数据
        for root, dirs, files in os.walk(full_path):
            rel_path = os.path.relpath(root, full_path)
            if root.endswith(bias_type):
                print(f"找到匹配的文件夹: {root}")
                for file in files:
                    if file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        key = f"{bias_type}_{rel_path}"
                        if key not in file_paths_dict:
                            file_paths_dict[key] = []
                        file_paths_dict[key].append(file_path)
                        print(f"找到CSV文件: {file_path}")
    
    print(f"\n总共找到 {len(file_paths_dict)} 个数据组")
    return file_paths_dict

def prepare_data(file_paths_dict):
    """
    读取所有dataframe，按bias类型和程度分组
    
    Args:
        file_paths_dict (dict): 包含文件路径的字典
        
    Returns:
        dict: 包含所有dataframe的字典，按bias类型和程度分组
    """
    dataframes_dict = {}
    for key, file_paths in file_paths_dict.items():
        dfs = [pd.read_csv(fp) for fp in file_paths]
        dataframes_dict[key] = dfs
        print(f"成功读取 {key}: {len(dfs)} 个dataframe")
    return dataframes_dict

def evaluate_data_get(data):
    return data['cate'],data['y'],data['T']

def data_loader(data, random_seed=42, remove_outliers=False, lower_pct=1, upper_pct=99):
    """
    将dataframe转换为numpy数组
    并且将数据集分为训练集、测试集、验证集（Bank数据集专用）
    
    Args:
        data: 原始数据DataFrame
        random_seed: 随机种子
        remove_outliers: 是否去除极端值（默认False）
        lower_pct: 下界百分位数
        upper_pct: 上界百分位数
    
    Returns:
        X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test, processed_data
    """
    raw_data = data.copy()
    
    # 根据 remove_outliers 参数决定是否去极端值
    if remove_outliers:
        treated = raw_data[raw_data['T'] == 1]
        control = raw_data[raw_data['T'] == 0]
        
        t_lower = treated['y'].quantile(lower_pct / 100)
        t_upper = treated['y'].quantile(upper_pct / 100)
        treated = treated[(treated['y'] >= t_lower) & (treated['y'] <= t_upper)]
        
        c_lower = control['y'].quantile(lower_pct / 100)
        c_upper = control['y'].quantile(upper_pct / 100)
        control = control[(control['y'] >= c_lower) & (control['y'] <= c_upper)]
        
        raw_data = pd.concat([treated, control]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    X = raw_data.drop(columns=['y','T','gamma_0','gamma_1','cate'])
    y = raw_data['y']
    T = raw_data['T']
   
    # Bank 数据集用 shuffle=True
    X_temp, X_test, y_temp, y_test, T_temp, T_test = train_test_split(X, y, T, test_size=0.3, random_state=random_seed, shuffle=True)
    X_train, X_val, y_train, y_val, T_train, T_val = train_test_split(X_temp, y_temp, T_temp, test_size=0.3, random_state=random_seed, shuffle=True)
    
    return X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test, raw_data



def calculate_metrics_for_multiple_runs(cate_results_df, df, run_times=10, k_list=None):
    """
    只处理多次运行（如 S_learner_run_1, S_learner_run_2, ...）的列。
    如果没有这些列则跳过该模型。
    """
    models = ['S_learner', 'T_learner', 'X_learner', 'R_learner', 'U_learner', 'DR_learner', 'RA_learner', 'CFR_learner', 'TAR_learner', 'Dragon_learner', 'True_uplift']
    results = {}

    for model in models:
        model_results = {}
        # 为每个k值存储PEHE和ATE error
        pehe_k_dict = {k: [] for k in k_list}
        ate_error_k_dict = {k: [] for k in k_list}
        # 不同k的指标
        qini_k_dict = {k: [] for k in k_list}
        uplift_k_dict = {k: [] for k in k_list}
        
        auuc_k_dict = {k: [] for k in k_list}

        for run in range(1, run_times + 1):
            col_name = f'{model}_run_{run}'
            if col_name in cate_results_df.columns:
                X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test, processed_data = data_loader(df, random_seed=(run-1)+42)
                test_indices = X_test.index
                cate_true_test = processed_data.loc[test_indices, 'cate'].values
                y_true_test = processed_data.loc[test_indices, 'y'].values
                T_true_test = processed_data.loc[test_indices, 'T'].values
                cate_pred = cate_results_df[col_name].values

                # 为每个k值计算PEHE和ATE error（不需要排序）
                for k in k_list:
                    if k == 1.0:
                        # 使用全部样本
                        k_samples = len(cate_pred)
                    else:
                        # 取前k%的样本（按索引顺序，不排序）
                        k_samples = int(len(cate_pred) * k)
                    
                    # 取前k_samples个样本
                    cate_pred_k = cate_pred[:k_samples]
                    cate_true_k = cate_true_test[:k_samples]
                    
                    # PEHE for k
                    pehe_val = pehe(cate_pred_k, cate_true_k, k=k)
                    pehe_k_dict[k].append(pehe_val)
                    
                    # ATE error for k
                    ate_error_val = ate_error(cate_pred_k, cate_true_k, k=k)
                    ate_error_k_dict[k].append(ate_error_val)
                
                # 不同k的Qini和Uplift（需要排序）
                for k in k_list:
                    qini_val = qini_coefficient_continuous_unscaled(y_true_test, cate_pred, T_true_test, k=k)
                    qini_k_dict[k].append(qini_val)
                    uplift_val = uplift_at_k_continuous(y_true_test, cate_pred, T_true_test, 'overall', k=k)
                    uplift_k_dict[k].append(uplift_val)
                
                # AUUC指标（现在也按k值计算）
                for k in k_list:
                    auuc_val = AUUC_continuous(y_true_test, cate_pred, T_true_test, k=k)
                    auuc_k_dict[k].append(auuc_val)

        # 只要有一次运行结果就写入，否则跳过该模型
        # 检查是否有任何运行结果（使用k_list中的第一个值）
        if pehe_k_dict[k_list[0]]:  # 检查是否有任何运行结果
            # 为每个k值存储PEHE和ATE error的结果
            for k in k_list:
                model_results[f'PEHE_k{k}_mean'] = np.mean(pehe_k_dict[k])
                model_results[f'PEHE_k{k}_std'] = np.std(pehe_k_dict[k])
                model_results[f'ATE_k{k}_mean'] = np.mean(ate_error_k_dict[k])  
                model_results[f'ATE_k{k}_std'] = np.std(ate_error_k_dict[k]) 
            
            # 不同k的Qini和Uplift
            for k in k_list:
                model_results[f'Qini_k{k}_mean'] = np.mean(qini_k_dict[k])
                model_results[f'Qini_k{k}_std'] = np.std(qini_k_dict[k])
                model_results[f'Uplift_k{k}_mean'] = np.mean(uplift_k_dict[k])
                model_results[f'Uplift_k{k}_std'] = np.std(uplift_k_dict[k])
            
            # AUUC指标
            for k in k_list:
                model_results[f'AUUC_k{k}_mean'] = np.mean(auuc_k_dict[k])
                model_results[f'AUUC_k{k}_std'] = np.std(auuc_k_dict[k])
            
            results[model] = model_results
        else:
            # 没有任何运行结果则不写入该模型
            continue

    # === 新增：用真实cate算理论最优指标 ===
    # 只需一遍（不多次），用和测试集一样的分法
    X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test, processed_data = data_loader(df, random_seed=42)
    test_indices = X_test.index
    cate_true_test = processed_data.loc[test_indices, 'cate'].values
    y_true_test = processed_data.loc[test_indices, 'y'].values
    T_true_test = processed_data.loc[test_indices, 'T'].values
    # 真实uplift就是cate_true_test
    true_uplift_results = {}
    
    # 为每个k值计算真实指标
    for k in k_list:
        if k == 1.0:
            k_samples = len(cate_true_test)
        else:
            k_samples = int(len(cate_true_test) * k)
        
        cate_true_k = cate_true_test[:k_samples]
        
        # PEHE（真实uplift和自己比为0）
        true_uplift_results[f'PEHE_k{k}_mean'] = 0.0
        true_uplift_results[f'PEHE_k{k}_std'] = 0.0
        # ATE error（真实uplift和自己比为0）
        true_uplift_results[f'ATE_k{k}_mean'] = 0.0  # 重命名为ATE
        true_uplift_results[f'ATE_k{k}_std'] = 0.0    # 重命名为ATE
    
    # === True_uplift 部分也做同样处理 ===
    for k in k_list:
        true_uplift_results[f'Qini_k{k}_mean'] = qini_coefficient_continuous_unscaled(y_true_test, cate_true_test, T_true_test, k=k)
        true_uplift_results[f'Qini_k{k}_std'] = 0.0
        true_uplift_results[f'Uplift_k{k}_mean'] = uplift_at_k_continuous(y_true_test, cate_true_test, T_true_test, 'overall', k=k)
        true_uplift_results[f'Uplift_k{k}_std'] = 0.0
        # AUUC指标（现在也按k值计算）
        true_uplift_results[f'AUUC_k{k}_mean'] = AUUC_continuous(y_true_test, cate_true_test, T_true_test, k=k)
        true_uplift_results[f'AUUC_k{k}_std'] = 0.0
    results['True_uplift'] = true_uplift_results

    return results

def create_results_dataframe(results, models, metrics):
    """
    创建包含平均值和标准差的DataFrame
    
    Args:
        results: 包含各模型结果的字典
        models: 模型列表
        metrics: 指标列表
    
    Returns:
        tuple: (平均值DataFrame, 标准差DataFrame)
    """
    # 创建平均值DataFrame
    mean_data = []
    std_data = []
    
    for metric in metrics:
        row_mean = {'metric': metric}
        row_std = {'metric': metric}
        
        for model in models:
            if model in results:
                row_mean[model] = results[model][f'{metric}_mean']
                row_std[model] = results[model][f'{metric}_std']
            else:
                row_mean[model] = np.nan
                row_std[model] = np.nan
        
        mean_data.append(row_mean)
        std_data.append(row_std)
    
    mean_df = pd.DataFrame(mean_data)
    std_df = pd.DataFrame(std_data)
    
    return mean_df, std_df

def create_mean_std_str_dataframe(results, models, metrics):
    """
    创建包含 mean±std 字符串的 DataFrame
    Args:
        results: 包含各模型结果的字典
        models: 模型列表
        metrics: 指标列表
    Returns:
        DataFrame: 每个单元格为 mean±std 字符串
    """
    data = []
    for metric in metrics:
        row = {'metric': metric}
        for model in models:
            if model in results:
                mean = results[model][f'{metric}_mean']
                std = results[model][f'{metric}_std']
                row[model] = f"{mean:.4f}±{std:.4f}"
            else:
                row[model] = ''
        data.append(row)
    df = pd.DataFrame(data)
    return df

def create_new_format_dataframe(all_results_dict, metric_type='PEHE', k_value=None):
    """
    创建新格式的DataFrame，横轴为不同bias和程度，纵轴为模型名称
    
    Args:
        all_results_dict: 包含所有数据集结果的字典 {dataset_key: results}
        metric_type: 指标类型 ('PEHE', 'ATE_error', 'Qini', 'Uplift')
        k_value: k值 (0.1, 0.3, 0.5, 1.0)，所有指标现在都需要k值
    
    Returns:
        DataFrame: 新格式的表格
    """
    models = ['S_learner', 'T_learner', 'X_learner', 'R_learner', 'U_learner', 'DR_learner', 'RA_learner', 'CFR_learner', 'TAR_learner', 'Dragon_learner', 'True_uplift']
    
    # 构建列名（横轴：不同bias和程度）
    columns = []
    for dataset_key in all_results_dict.keys():
        columns.append(dataset_key)
    
    # 构建数据
    data = []
    for model in models:
        row = {'Model': model}
        for dataset_key in all_results_dict.keys():
            results = all_results_dict[dataset_key]
            if model in results:
                if k_value is None:
                    raise ValueError(f"k_value must be provided for all metrics")
                
                # 所有指标现在都使用k值
                mean_val = results[model][f'{metric_type}_k{k_value}_mean']
                std_val = results[model][f'{metric_type}_k{k_value}_std']
                
                row[dataset_key] = f"{mean_val:.4f}±{std_val:.4f}"
            else:
                row[dataset_key] = ''
        data.append(row)
    
    df = pd.DataFrame(data)
    return df

def generate_metric_table(all_results_dict, metric_type='PEHE', k_value=None, save_path=None):
    """
    根据参数生成指定指标的表格
    
    Args:
        all_results_dict: 包含所有数据集结果的字典
        metric_type: 指标类型 ('PEHE', 'ATE_error', 'Qini', 'Uplift')
        k_value: k值，所有指标现在都需要k值
        save_path: 保存路径，如果为None则不保存
    
    Returns:
        DataFrame: 生成的表格
    """
    if k_value is None:
        raise ValueError(f"k_value must be provided for all metrics")
    
    df = create_new_format_dataframe(all_results_dict, metric_type=metric_type, k_value=k_value)
    
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 根据k值生成文件名
        if k_value == 1.0:
            filename = f"{metric_type.lower()}_all.csv"
        else:
            filename = f"{metric_type.lower()}_k{k_value}.csv"
        
        filepath = os.path.join(save_path, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"{metric_type}表格已保存到: {filepath}")
    
    return df

def get_cate_results_path(bias_type, degree_info, dataset_name='Bank'):
    """
    根据bias类型和程度信息获取对应的cate结果文件路径
    
    Args:
        bias_type (str): bias类型 (measurement, selection, spillover, hidden)
        degree_info (str): 程度信息
        dataset_name (str): 数据集名称 (Hillstrom 或 Bank)
        
    Returns:
        str: cate结果文件路径
    """
    # 拼接路径: output_data/dataset_name/bias_type
    base_path = os.path.join(r"output_data", dataset_name, bias_type)
    
    # 生成文件名 - degree_info 中可能已经包含 cate_results_ 前缀
    if degree_info.startswith('cate_results_'):
        # 如果已经有前缀，直接使用
        filename = f"{degree_info}.csv"
    else:
        # 如果没有前缀，添加前缀
        filename = f"cate_results_{bias_type}_{degree_info}.csv"
    
    return os.path.join(base_path, filename)

if __name__ == "__main__":
    # 设置是否读取所有数据
    all_data = True  # 设置为True来读取所有数据
    # dataset_name = 'Hillstrom'
    dataset_name = 'Criteo'

    # 设置要生成的指标类型和k值
    target_metric = 'PEHE'  # 可选: 'PEHE', 'ATE', 'AUUC', 'Qini', 'Uplift'
    target_k = 0.3  # 所有指标现在都需要k值
    
    file_paths_dict = get_raw_data_path(folder_path=r"data", dataset_name=dataset_name, bias_type="all", all_data=all_data)
    dataframes_dict = prepare_data(file_paths_dict)
    
    models = ['S_learner', 'T_learner', 'X_learner', 'R_learner', 'U_learner', 'DR_learner', 'RA_learner', 'TAR_learner', 'Dragon_learner']
    k_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
    metrics = ['PEHE', 'ATE', 'AUUC', 'Qini', 'Uplift']  
    # 存储所有数据集的结果
    all_results_dict = {}
    
    # 遍历所有dataframe
    for key, dfs in dataframes_dict.items():
        print(f"\n=== 处理数据集: {key} ===")
        
        # 解析key获取bias类型和程度信息
        key_parts = key.split('_')
        bias_type = key_parts[0]  # measurement, selection, spillover
        
        # 从key中提取程度信息
        if len(key_parts) > 1:
            degree_info = '_'.join(key_parts[1:])  # 获取所有程度信息
        else:
            degree_info = "default"
        
        # 遍历该key下的所有dataframe
        for df_idx, df in enumerate(dfs):
            print(f"处理第 {df_idx + 1} 个dataframe...")
            
            # 获取对应的cate结果文件路径
            cate_results_path = get_cate_results_path(bias_type, degree_info, dataset_name)
            
            # 检查cate结果文件是否存在
            if not os.path.exists(cate_results_path):
                print(f"警告: cate结果文件不存在: {cate_results_path}")
                continue
            
            try:
                # 读取十次运行的cate预测结果
                cate_results_df = pd.read_csv(cate_results_path)
                
                # 计算多次运行的指标平均值和标准差
                results = calculate_metrics_for_multiple_runs(cate_results_df, df, run_times=10, k_list=k_list)
                
                # 存储结果
                all_results_dict[key] = results
                
                # 根据指标类型确定保存路径
                base_save_path = fr"result/{dataset_name}"
                if target_metric == 'PEHE':
                    save_path = os.path.join(base_save_path, "PEHE")
                elif target_metric == 'ATE':
                    save_path = os.path.join(base_save_path, "ATE")
                elif target_metric == 'AUUC':
                    save_path = os.path.join(base_save_path, "AUUC")
                elif target_metric == 'Qini':
                    save_path = os.path.join(base_save_path, "Qini")
                elif target_metric == 'Uplift':
                    save_path = os.path.join(base_save_path, "Uplift")
                else:
                    save_path = os.path.join(base_save_path, "other")
                
                # 根据target_metric生成对应的表格
                target_df = generate_metric_table(all_results_dict, metric_type=target_metric, k_value=target_k, save_path=save_path)
                
                print(f"目标指标表格已生成: {target_metric}")
                print(f"k值: {target_k}")
                print(f"保存路径: {save_path}")
                
                # 打印结果摘要
                print(f"\n=== {key} 指标评估结果摘要 ===")
                for model in models:
                    if model in results:
                        print(f"\n{model}:")
                        for k in k_list:
                            print(f"  PEHE@K={k*100:.0f}%: {results[model][f'PEHE_k{k}_mean']:.4f} ± {results[model][f'PEHE_k{k}_std']:.4f}")
                            print(f"  ATE@K={k*100:.0f}%: {results[model][f'ATE_k{k}_mean']:.4f} ± {results[model][f'ATE_k{k}_std']:.4f}")
                            print(f"  AUUC@K={k*100:.0f}%: {results[model][f'AUUC_k{k}_mean']:.4f} ± {results[model][f'AUUC_k{k}_std']:.4f}")
                            print(f"  Qini@K={k*100:.0f}%: {results[model][f'Qini_k{k}_mean']:.4f} ± {results[model][f'Qini_k{k}_std']:.4f}")
                            print(f"  Uplift@K={k*100:.0f}%: {results[model][f'Uplift_k{k}_mean']:.4f} ± {results[model][f'Uplift_k{k}_std']:.4f}")
                
            except Exception as e:
                print(f"处理 {key} 时发生错误: {str(e)}")
                continue
        
        
        

    
       
    





