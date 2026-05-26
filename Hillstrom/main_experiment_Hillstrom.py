from meta_learner import *
from Net_model.main_CFR import *
from Net_model.main_TAR import *
from Dragonnet.main_Dragon import *
import pandas as pd
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from Net_optimization_Hillstrom import *
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_data_path(folder_path=r"data/Hillstrom", bias_type="selection", all_data=False):
    """
    读取指定文件夹下的所有CSV文件路径，支持all_data模式自动遍历所有bias和程度
    
    Args:
        folder_path (str): 数据文件夹路径
        bias_type (str): 偏差类型，用于筛选特定的子文件夹
        all_data (bool): 是否读取所有数据，如果为True则忽略bias_type参数
    
    Returns:
        dict: 包含所有CSV文件路径的字典，按bias类型和程度分组
    """
    import re
    file_paths_dict = {}
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

def data_loader(data, random_seed=42):
    """
    将dataframe转换为numpy数组
    并且将数据集分为训练集、测试集、验证集
    """
    raw_data = data.copy()
    X = raw_data.drop(columns=['y','T','gamma_0','gamma_1','cate'])
    y = data['y']
    T = data['T']
   
    # shuffle=False 保证数据集的顺序不变
    X_temp, X_test, y_temp, y_test, T_temp, T_test = train_test_split(X, y, T, test_size=0.3, random_state=random_seed,shuffle=False)
    X_train, X_val, y_train, y_val, T_train, T_val = train_test_split(X_temp, y_temp, T_temp, test_size=0.3, random_state=random_seed,shuffle=False)
    
    return X_train, X_val, y_train, y_val, T_train, T_val, X_test, y_test, T_test


if __name__ == "__main__":
    # meta-learner的参数
    xgb_opt_early_stopping_rounds = 12
    y_opt_early_stopping_rounds = 12

    n_trials_xgb = 50    
    n_trials_lr = 50    
    n_trials_propensity = 50 
    trial_patience = 8

    run_times = 10  # PEHE指标运行次数

    # Net模型的参数
    net_n_runs = 2             
    opt_epochs = 50            
    train_epochs = 100         
   
    train_opt_patience = 5     
    train_patience = 10        
    search_early_stop_patience = 3  
    
    
   
    
    # 设置是否读取所有数据
    all_data = False  # 设置为True来读取所有数据
    
    file_paths_dict = get_data_path(folder_path=r"data/Hillstrom", bias_type="spillover", all_data=all_data)
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
            CFR_cate_list = []
            TAR_cate_list = []
            Dragon_cate_list = []
            

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
                
                # S-learner cate
                S_model_cate = S_learner(**S_params)
                S_cate_list.append(S_model_cate)
                
                # T-learner cate
                T_model_cate = T_learner(**T_params)
                T_cate_list.append(T_model_cate)
                
                # X-learner cate
                X_model_cate = X_learner(**X_params)
                X_cate_list.append(X_model_cate)
                
                # R-learner cate
                R_model_cate = R_learner(**R_params)
                R_cate_list.append(R_model_cate)
                
                # U-learner cate
                U_model_cate = U_learner(**U_params)
                U_cate_list.append(U_model_cate)
                
                # DR-learner cate
                DR_model_cate = DR_learner(**DR_params)
                DR_cate_list.append(DR_model_cate)
                
                # RA-learner cate
                RA_model_cate = RA_learner(**RA_params)
                RA_cate_list.append(RA_model_cate)
                
                # # ========== 神经网络模型训练和预测 ==========
                print(f"开始训练神经网络模型...")
                
                # 准备神经网络训练数据
                X_train_tensor = torch.FloatTensor(X_train.values).to(device)
                X_val_tensor = torch.FloatTensor(X_val.values).to(device)
                X_test_tensor = torch.FloatTensor(X_test.values).to(device)
                y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
                y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
                y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1).to(device)
                T_train_tensor = torch.FloatTensor(T_train.values).reshape(-1, 1).to(device)
                T_val_tensor = torch.FloatTensor(T_val.values).reshape(-1, 1).to(device)
                T_test_tensor = torch.FloatTensor(T_test.values).reshape(-1, 1).to(device)
                
                # 计算处理概率
                p_t = T_train.mean()
                
                # # 1. CFR模型调参和训练
                # print("开始CFR模型调参...")
                
                # # CFR调参
                # cfr_best_params = optimize_cfr(
                #     X_train=X_train.values,
                #     X_val=X_val.values,
                #     y_train=y_train,
                #     y_val=y_val,
                #     T_train=T_train,
                #     T_val=T_val,
                #     n_runs=net_n_runs,
                #     epochs=opt_epochs,
                #     train_patience=train_opt_patience,
                #     search_patience=search_early_stop_patience
                # )
                
                # print("CFR调参完成，开始训练...")
                # # CFR训练和预测
                # cfr_results = train_cfr(
                #     X_train=X_train.values,
                #     X_val=X_val.values,
                #     y_train=y_train,
                #     y_val=y_val,
                #     T_train=T_train,
                #     T_val=T_val,
                #     X_test=X_test.values,  # 使用测试集进行预测
                #     y_test=y_test,
                #     T_test=T_test,
                #     best_params=cfr_best_params,
                #     epochs=train_epochs,
                #     train_patience=train_patience
                # )
                
                # # 提取CFR的CATE预测结果
                # CFR_cate = cfr_results['CFR'].values
                # CFR_cate_list.append(CFR_cate)
                
                # 2. TAR模型调参和训练
                print("开始TAR模型调参...")
                    
                
                # TAR调参
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
                
                print("TAR调参完成，开始训练...")
                # TAR训练和预测
                tar_results = train_tar(
                    X_train=X_train.values,
                    X_val=X_val.values,
                    y_train=y_train,
                    y_val=y_val,
                    T_train=T_train,
                    T_val=T_val,
                    X_test=X_test.values,  # 使用测试集进行预测
                    y_test=y_test,
                    T_test=T_test,
                    best_params=tar_best_params,
                    epochs=train_epochs,
                    train_patience=train_patience
                )
                
                # 提取TAR的CATE预测结果
                TAR_cate = tar_results['TAR'].values
                TAR_cate_list.append(TAR_cate)
                
                # 3. DragonNet模型调参和训练
                print("开始DragonNet模型调参...")
                
                
                # DragonNet调参
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
                
                print("DragonNet调参完成，开始训练...")
                # DragonNet训练和预测
                dragon_results = train_dragonnet(
                    X_train=X_train.values,
                    X_val=X_val.values,
                    y_train=y_train,
                    y_val=y_val,
                    T_train=T_train,
                    T_val=T_val,
                    X_test=X_test.values,  # 使用测试集进行预测
                    y_test=y_test,
                    T_test=T_test,
                    best_params=dragon_best_params,
                    epochs=train_epochs,
                    train_patience=train_patience
                )
                
                # 提取DragonNet的CATE预测结果
                Dragon_cate = dragon_results['Dragon'].values
                Dragon_cate_list.append(Dragon_cate)
                
                print(f"神经网络模型训练完成")
            
            # 保存所有cate预测结果到CSV文件
            # 创建包含所有cate预测结果的DataFrame
            all_cate_results = {}
            
            # 为每个learner添加cate预测结果
            for i in range(run_times):
                all_cate_results[f'S_learner_run_{i+1}'] = S_cate_list[i]
                all_cate_results[f'T_learner_run_{i+1}'] = T_cate_list[i]
                all_cate_results[f'X_learner_run_{i+1}'] = X_cate_list[i]
                all_cate_results[f'R_learner_run_{i+1}'] = R_cate_list[i]
                all_cate_results[f'U_learner_run_{i+1}'] = U_cate_list[i]
                all_cate_results[f'DR_learner_run_{i+1}'] = DR_cate_list[i]
                all_cate_results[f'RA_learner_run_{i+1}'] = RA_cate_list[i]
                
                # 添加神经网络模型的cate预测结果
                # all_cate_results[f'CFR_learner_run_{i+1}'] = CFR_cate_list[i]
                all_cate_results[f'TAR_learner_run_{i+1}'] = TAR_cate_list[i]
                all_cate_results[f'Dragon_learner_run_{i+1}'] = Dragon_cate_list[i]
            
            # 添加真实的cate值
            
            
            # 转换为DataFrame
            cate_results_df = pd.DataFrame(all_cate_results)
            
            # 打印结果
            print(f"\n=== CATE预测结果汇总 ===")
            print(f"总共保存了 {run_times} 次运行的cate预测结果")
            print(f"包含7个meta-learner模型和3个神经网络模型")
            
            
            # 保存结果
            # 解析key获取bias类型和程度信息
            key_parts = key.split('_')
            bias_type = key_parts[0]  # measurement, selection, spillover
            
            # 从key中提取程度信息
            if len(key_parts) > 1:
                degree_info = '_'.join(key_parts[1:])  # 获取所有程度信息
            else:
                degree_info = "default"
            
            # 根据bias类型确定保存路径
            if bias_type == 'measurement':
                save_path = r"output_data/Hillstrom/measurement"
            elif bias_type == 'selection':
                save_path = r"output_data/Hillstrom/selection"
            elif bias_type == 'spillover':
                save_path = r"output_data/Hillstrom/spillover"
            else:
                save_path = r"output_data/Hillstrom/hidden"
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # 生成文件名，包含bias类型和程度信息
            filename = f"cate_results_{bias_type}_{degree_info}.csv"
            cate_results_df.to_csv(os.path.join(save_path, filename), index=False)
            
            print(f"结果已保存到: {os.path.join(save_path, filename)}")
        
       
        
   




 








 
