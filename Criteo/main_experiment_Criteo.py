from meta_learner import *
from Net_model.main_CFR import *
from Net_model.main_TAR import *
from Dragonnet.main_Dragon import *
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from Net_optimization_Criteo import *
from metric import *
import warnings
import random
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    """固定所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

dataset_name='Criteo'

def get_data_path(folder_path=r"data", dataset_name=dataset_name, bias_type="selection", all_data=False):
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
    import re
    file_paths_dict = {}
    
    # 拼接数据集名称到路径中
    folder_path = os.path.join(folder_path, dataset_name)
    
    if not os.path.exists(folder_path):
        print(f"错误: 目录 '{folder_path}' 不存在")
        return file_paths_dict

    if all_data:
        # 遍历所有 bias 文件夹
        for bias_type in os.listdir(folder_path):
            bias_path = os.path.join(folder_path, bias_type)
            if not os.path.isdir(bias_path):
                continue
            # degree 分组
            degree_dict = {}
            for file in os.listdir(bias_path):
                if file.endswith('.csv'):
                    # 匹配“程度”部分（去掉最后的 _数字.csv）
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
        # 只读取指定bias_type的数据，并按degree分组
        for root, dirs, files in os.walk(folder_path):
            rel_path = os.path.relpath(root, folder_path)
            if root.endswith(bias_type):
                print(f"找到匹配的文件夹: {root}")
                degree_dict = {}
                for file in files:
                    if file.endswith('.csv'):
                        match = re.match(r'(.+)_\d+\.csv$', file)
                        if match:
                            degree_prefix = match.group(1)
                        else:
                            degree_prefix = file.split('.csv')[0]
                        if degree_prefix not in degree_dict:
                            degree_dict[degree_prefix] = []
                        degree_dict[degree_prefix].append(os.path.join(root, file))
                for degree, files in degree_dict.items():
                    key = f"{bias_type}_{degree}"
                    file_paths_dict[key] = files
                    print(f"找到: {key} -> {len(files)} 个文件")
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

def data_loader(data, random_seed=42, remove_outliers=False, lower_pct=1, upper_pct=99):
    """
    将dataframe转换为numpy数组
    并且将数据集分为训练集、测试集、验证集
    
    Args:
        data: 原始数据DataFrame
        random_seed: 随机种子
        remove_outliers: 是否去除y的极端值（分treatment/control两组分别去除）
        lower_pct: 下界百分位数（默认1%）
        upper_pct: 上界百分位数（默认99%）
    """
    raw_data = data.copy()
    
    # 分 treatment/control 两组分别去除各自的极端值，避免引入组间偏差
    if remove_outliers:
        treated = raw_data[raw_data['T'] == 1]
        control = raw_data[raw_data['T'] == 0]
        
        # treatment 组去自己的极端值
        t_lower = treated['y'].quantile(lower_pct / 100)
        t_upper = treated['y'].quantile(upper_pct / 100)
        treated = treated[(treated['y'] >= t_lower) & (treated['y'] <= t_upper)]
        
        # control 组去自己的极端值
        c_lower = control['y'].quantile(lower_pct / 100)
        c_upper = control['y'].quantile(upper_pct / 100)
        control = control[(control['y'] >= c_lower) & (control['y'] <= c_upper)]
        
        # 合并回来，并用固定种子打乱顺序（避免前半全是treatment后半全是control）
        raw_data = pd.concat([treated, control]).sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    X = raw_data.drop(columns=['y','T','gamma_0','gamma_1','cate'])
    y = raw_data['y']
    T = raw_data['T']
   
    # shuffle=True + random_state 确保每次run数据划分不同但可复现
    X_temp, X_test, y_temp, y_test, T_temp, T_test = train_test_split(X, y, T, test_size=0.3, random_state=random_seed, shuffle=True)
    X_train, X_val, y_train, y_val, T_train, T_val = train_test_split(X_temp, y_temp, T_temp, test_size=0.3, random_state=random_seed, shuffle=True)
    
    return X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test


if __name__ == "__main__":
    # meta-learner的参数 - 扩大调参空间以提高稳定性
    xgb_opt_early_stopping_rounds = 3    # 原始5
    y_opt_early_stopping_rounds = 3      # 原始5

    n_trials_xgb = 8         # 原始8
    n_trials_lr = 8          # 原始8
    n_trials_propensity = 8  # 原始8
    trial_patience = 3        # 原始3

    run_times = 5  # PEHE指标运行次数

    # 模型运行开关
    run_meta_learners = True    # 跑 meta-learner (S/T/X/R/U/DR/RA)
    run_tarnet = True          # 跑 TARNet
    run_dragonnet = True        # 跑 DragonNet（调好后设为True）

  
    # Net model parameters 
    net_n_runs = 1              # 每个参数组合运行次数
    opt_epochs = 50             # 调参阶段训练轮数
    train_epochs = 2000          # 最终训练轮数
   
    train_opt_patience = 10      # 调参阶段早停耐心
    train_patience = 35         # 最终训练早停耐心
    search_early_stop_patience = 8  # 搜索早停耐心
    
    
    
   
    
    # 设置是否读取所有数据
    all_data = True  # 设置为True来读取所有数据
    
    file_paths_dict = get_data_path(folder_path=r"data", dataset_name=dataset_name, bias_type="measurement", all_data=all_data)
    dataframes_dict = prepare_data(file_paths_dict)
    
    # 遍历所有dataframe
    for key, dfs in dataframes_dict.items():
        print(f"\n=== 处理数据集: {key} ===")
        # 遍历该key下的所有dataframe
        for df_idx, df in enumerate(dfs):
            print(f"处理第 {df_idx + 1} 个dataframe...")
            
            # 存储所有运行的结果
            X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test = data_loader(df)            
            
            # 初始化存储每次运行cate的列表
            S_cate_list = []
            T_cate_list = []
            X_cate_list = []
            R_cate_list = []
            U_cate_list = []
            DR_cate_list = []
            RA_cate_list = []
            
            # 神经网络模型的cate列表
            # CFR_cate_list = []
            TAR_cate_list = []
            Dragon_cate_list = []
            

            print(f"\n========== 开始神经网络模型超参数优化 ==========")
            
            # 固定随机种子，确保超参数优化可复现
            set_seed(42)
            
            # TAR模型调参
            if run_tarnet:
                print("开始TAR模型调参...")
                tar_best_params = optimize_tar(
                    X_train=X_train.values,
                    X_val=X_val.values,
                    y_train=y_train,
                    y_val=y_val,
                    T_train=T_train,
                    T_val=T_val,
                    n_runs=net_n_runs,
                    epochs=opt_epochs,
                    train_patience=train_opt_patience,
                    search_patience=search_early_stop_patience
                )
                print(f"TAR调参完成，最佳参数: {tar_best_params}")
            
            # DragonNet模型调参
            if run_dragonnet:
                print("\n开始DragonNet模型调参...")
                dragon_best_params = optimize_dragonnet(
                    X_train=X_train.values,
                    X_val=X_val.values,
                    y_train=y_train,
                    y_val=y_val,
                    T_train=T_train,
                    T_val=T_val,
                    n_runs=net_n_runs,
                    epochs=opt_epochs,
                    train_patience=train_opt_patience,
                    search_patience=search_early_stop_patience
                )
                print(f"DragonNet调参完成，最佳参数: {dragon_best_params}")
            print(f"========== 神经网络模型超参数优化完成 ==========\n")
            

            S_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test,'X_val': X_val, 'y_val': y_val,
                'n_trials_xgb': n_trials_xgb,'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,'trial_patience': trial_patience
            }
            T_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test,'X_val': X_val, 'y_val': y_val,
                'n_trials_xgb': n_trials_xgb,'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,
                'trial_patience': trial_patience
            }
            X_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test, 'X_val': X_val, 'T_val': T_val, 'y_val': y_val, 
                'n_trials_xgb': n_trials_xgb, 'n_trials_propensity': n_trials_propensity, 
                'n_trials_lr': n_trials_lr, 'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,
                'y_opt_early_stopping_rounds': y_opt_early_stopping_rounds,
                'trial_patience': trial_patience
            }
            R_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test, 'X_val': X_val, 'y_val': y_val, 
                'n_trials_xgb': n_trials_xgb, 'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,
                'trial_patience': trial_patience
            }
            U_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test, 'X_val': X_val, 'y_val': y_val, 
                'n_trials_xgb': n_trials_xgb, 'n_trials_lr': n_trials_lr,
                'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,
                'trial_patience': trial_patience
            }
            DR_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test, 'X_val': X_val, 'T_val': T_val,'y_val': y_val,
                'n_trials_xgb': n_trials_xgb, 'n_trials_lr': n_trials_lr,
                'n_trials_propensity': n_trials_propensity,
                'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,
                'y_opt_early_stopping_rounds': y_opt_early_stopping_rounds,
                'trial_patience': trial_patience
            }
            RA_params = {
                'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                'X_test': X_test, 'X_val': X_val, 'T_val': T_val, 'y_val': y_val,
                'n_trials_xgb': n_trials_xgb, 'n_trials_propensity': n_trials_propensity,
                'xgb_opt_early_stopping_rounds': xgb_opt_early_stopping_rounds,
                'y_opt_early_stopping_rounds': y_opt_early_stopping_rounds,
                'trial_patience': trial_patience
            }
            
                 
            # 后续运行：保存每次的cate预测结果
            for run in range(run_times):
                print(f"运行第 {run + 1}/{run_times} 次（保存cate预测结果）...")
                
                # 固定随机种子，确保每次运行可复现
                set_seed(run + 42)
                
                # 每次重新划分数据（使用不同的随机种子）
                X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test = data_loader(df, random_seed=run+42)
                
                # 更新参数中的训练数据
                S_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'y_val': y_val
                })
                T_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'y_val': y_val
                })
                X_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'T_val': T_val, 'y_val': y_val
                })
                R_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'y_val': y_val
                })
                U_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'y_val': y_val
                })
                DR_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'T_val': T_val,'y_val': y_val
                })
                RA_params.update({
                    'X_train': X_train, 'T_train': T_train, 'y_train': y_train, 
                    'X_test': X_test, 'X_val': X_val, 'T_val': T_val, 'y_val': y_val
                })
                
                # Meta-learner cate
                if run_meta_learners:
                    S_model_cate = S_learner(**S_params)
                    S_cate_list.append(S_model_cate)
                    
                    T_model_cate = T_learner(**T_params)
                    T_cate_list.append(T_model_cate)
                    
                    X_model_cate = X_learner(**X_params)
                    X_cate_list.append(X_model_cate)
                    
                    R_model_cate = R_learner(**R_params)
                    R_cate_list.append(R_model_cate)
                    
                    U_model_cate = U_learner(**U_params)
                    U_cate_list.append(U_model_cate)
                    
                    DR_model_cate = DR_learner(**DR_params)
                    DR_cate_list.append(DR_model_cate)
                    
                    RA_model_cate = RA_learner(**RA_params)
                    RA_cate_list.append(RA_model_cate)
                
                # ========== 神经网络模型训练和预测（使用固定的最佳参数） ==========
                print(f"开始训练神经网络模型（使用已优化的参数）...")
                
                # 2. TAR模型训练和预测（使用固定的最佳参数）
                if run_tarnet:
                    print("开始TAR模型训练...")
                    tar_results = train_tar(
                        X_train=X_train.values,
                        X_val=X_val.values,
                        y_train=y_train,
                        y_val=y_val,
                        T_train=T_train,
                        T_val=T_val,
                        X_test=X_test.values,
                        y_test=y_test,
                        T_test=T_test,
                        best_params=tar_best_params,
                        epochs=train_epochs,
                        train_patience=train_patience
                    )
                    TAR_cate = tar_results['TAR'].values
                    TAR_cate_list.append(TAR_cate)
                
                # 3. DragonNet模型训练和预测（使用固定的最佳参数）
                if run_dragonnet:
                    print("开始DragonNet模型训练...")
                    dragon_results = train_dragonnet(
                        X_train=X_train.values,
                        X_val=X_val.values,
                        y_train=y_train,
                        y_val=y_val,
                        T_train=T_train,
                        T_val=T_val,
                        X_test=X_test.values,
                        y_test=y_test,
                        T_test=T_test,
                        best_params=dragon_best_params,
                        epochs=train_epochs,
                        train_patience=train_patience
                    )
                    Dragon_cate = dragon_results['Dragon'].values
                    Dragon_cate_list.append(Dragon_cate)
                
                print(f"神经网络模型训练完成")
            
            # 保存所有cate预测结果到CSV文件
            # 创建包含所有cate预测结果的DataFrame
            all_cate_results = {}
            
            # 为每个learner添加cate预测结果
            for i in range(run_times):
                if run_meta_learners:
                    all_cate_results[f'S_learner_run_{i+1}'] = S_cate_list[i]
                    all_cate_results[f'T_learner_run_{i+1}'] = T_cate_list[i]
                    all_cate_results[f'X_learner_run_{i+1}'] = X_cate_list[i]
                    all_cate_results[f'R_learner_run_{i+1}'] = R_cate_list[i]
                    all_cate_results[f'U_learner_run_{i+1}'] = U_cate_list[i]
                    all_cate_results[f'DR_learner_run_{i+1}'] = DR_cate_list[i]
                    all_cate_results[f'RA_learner_run_{i+1}'] = RA_cate_list[i]
                
                # 添加神经网络模型的cate预测结果
                if run_tarnet:
                    all_cate_results[f'TAR_learner_run_{i+1}'] = TAR_cate_list[i]
                if run_dragonnet:
                    all_cate_results[f'Dragon_learner_run_{i+1}'] = Dragon_cate_list[i]
            
            # 添加真实的cate值
            
            
            # 转换为DataFrame
            cate_results_df = pd.DataFrame(all_cate_results)
            
            # 打印结果
            print(f"\n=== CATE预测结果汇总 ===")
            
            
            
            # 保存结果
            # 解析key获取bias类型和程度信息
            key_parts = key.split('_')
            bias_type = key_parts[0]  # measurement, selection, spillover
            
            # 从key中提取程度信息
            if len(key_parts) > 1:
                degree_info = '_'.join(key_parts[1:])  # 获取所有程度信息
            else:
                degree_info = "default"
            
            # 根据数据集和bias类型确定保存路径
            save_path = os.path.join("output_data", dataset_name, bias_type)
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # 生成文件名，包含bias类型和程度信息
            filename = f"cate_results_{bias_type}_{degree_info}.csv"
            filepath = os.path.join(save_path, filename)
            
            # 如果只跑部分模型，尝试读取已有结果并合并
            if os.path.exists(filepath) and not (run_meta_learners and run_tarnet and run_dragonnet):
                existing_df = pd.read_csv(filepath)
                # 把新跑的列覆盖到已有结果中
                for col in cate_results_df.columns:
                    existing_df[col] = cate_results_df[col]
                existing_df.to_csv(filepath, index=False)
                print(f"结果已合并保存到: {filepath}")
            else:
                cate_results_df.to_csv(filepath, index=False)
                print(f"结果已保存到: {filepath}")
        
       
        
   




 








 
