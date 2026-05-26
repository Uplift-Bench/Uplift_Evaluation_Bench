
import torch
import numpy as np
import random
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')
import pandas as pd

def set_seed(seed=42):
    """固定所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import os
import csv
from datetime import datetime

def log_training_result(model_name, params, epochs_run, best_val_loss, cate_mean, cate_std, cate_min, cate_max, note=""):
    """记录训练结果到 training_log.csv"""
    log_file = "training_log.csv"
    file_exists = os.path.exists(log_file)
    
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['时间', '模型', 'learning_rate', 'hidden_dim', 'alpha', 'beta', 
                           'batch_size', 'epochs_run', 'best_val_loss', 
                           'cate_mean', 'cate_std', 'cate_min', 'cate_max', '备注'])
        
        writer.writerow([
            datetime.now().strftime('%m/%d %H:%M'),
            model_name,
            params.get('learning_rate', '-'),
            f"{params.get('rep_hidden_dim', params.get('shared_hidden', '-'))}/{params.get('hyp_hidden_dim', params.get('outcome_hidden', '-'))}",
            params.get('alpha', '-'),
            params.get('beta', '-'),
            params.get('batch_size', '-'),
            epochs_run,
            f"{best_val_loss:.4f}" if best_val_loss else '-',
            f"{cate_mean:.4f}",
            f"{cate_std:.4f}",
            f"{cate_min:.4f}",
            f"{cate_max:.4f}",
            note
        ])
    print(f"[LOG] 训练结果已记录到 {log_file}")

# 导入模型和训练函数
from Net_model.main_CFR import CFRNet, train_CFR
from Net_model.main_TAR import TARNet, train_TAR
from Dragonnet.main_Dragon import DragonNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_cfr(X_train: np.ndarray, 
                 X_val: np.ndarray, 
                 y_train: np.ndarray, 
                 y_val: np.ndarray, 
                 T_train: np.ndarray, 
                 T_val: np.ndarray,
                 n_runs: int = 5,
                 epochs: int = 100,
                 train_patience: int = 10,
                 search_patience: int = 20) -> Dict[str, Any]:
    """
    CFR模型精简网格搜索调参并返回最优参数
    
    推荐配置：只搜索最重要的参数
    - 高重要性参数（搜索）：alpha, rep_hidden_dim, hyp_hidden_dim
    - 中等重要性参数（固定）：batch_size, rep_layers, hyp_layers
    - 低重要性参数（固定）：dropout_rate, learning_rate, weight_decay等
    
    Args:
        n_runs: 每个参数组合运行的次数，用于取平均值
        train_patience: 训练早停耐心值，连续多少轮验证损失没有改善就停止
        search_patience: 搜索早停耐心值，连续多少轮没有改善就停止搜索
    
    Returns:
        best_params: 最优参数字典
    """
    import itertools
    
    # 精简参数搜索空间 - 只搜索最重要的参数
    param_grid = {
        'alpha': [10**(k/2) for k in range(-8, 5)],  
        'rep_hidden_dim': [50, 100, 200], 
        'hyp_hidden_dim': [50, 100, 200],  
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"CFR精简网格搜索开始 - 总参数组合数: {len(param_combinations)}")
    print(f"每个组合运行次数: {n_runs}")
    print(f"搜索参数: {list(param_grid.keys())}")
    print(f"搜索早停耐心值: {search_patience}")
    
    # 准备数据
    train_data = {
        'x': X_train.astype(np.float32),
        'y': y_train.values if hasattr(y_train, 'values') else y_train.astype(np.float32),
        't': T_train.values if hasattr(T_train, 'values') else T_train.astype(np.float32)
    }
    
    val_data = {
        'x': X_val.astype(np.float32),
        'y': y_val.values if hasattr(y_val, 'values') else y_val.astype(np.float32),
        't': T_val.values if hasattr(T_val, 'values') else T_val.astype(np.float32)
    }
    
    input_dim = X_train.shape[1]
    
    # 记录最佳参数
    best_avg_loss = float('inf')
    best_params = None
    no_improvement_count = 0  # 优化早停计数器
    
    def train_single_run(params_dict):
        """单次训练运行"""
        try:
            # 固定参数（优化后的经验值）
            fixed_params = {
                'batch_size': 500,           # 固定批次大小
                'rep_layers': 3,             # 固定表示层数量为3
                'hyp_layers': 3,             # 固定假设层数量为3
                'dropout_rate': 0.1,         # 固定dropout率
                'weight_init_std': 0.1,      # 固定权重初始化
                'learning_rate': 5e-4,       # 降低学习率，更稳定收敛
                'weight_decay': 1e-4,        # 固定权重衰减
                'lr_decay_rate': 0.97,       # 衰减更慢
                'lr_decay_steps': 200,       # 衰减间隔拉大
                'use_lr_scheduler': True,    # 固定使用学习率调度器
                'split_output': True,        # 固定输出分离
                'varsel': False,             # 固定变量选择
                'batch_norm': True,          # 固定批量归一化
                'activation': 'relu',        # 固定激活函数
                'imb_type': 'mmd_lin',       # 固定不平衡类型
                'loss_type': 'l2',           # 固定损失类型
                'reweight_sample': False,    # 固定样本重加权
                'use_p_correction': False    # 固定处理概率校正
            }
            
            # 合并参数
            full_params = {**fixed_params, **params_dict}
            
            model = CFRNet(
                input_dim=input_dim,
                rep_hidden_dim=full_params['rep_hidden_dim'],
                hyp_hidden_dim=full_params['hyp_hidden_dim'],
                rep_layers=full_params['rep_layers'],
                hyp_layers=full_params['hyp_layers'],
                split_output=full_params['split_output'],
                varsel=full_params['varsel'],
                batch_norm=full_params['batch_norm'],
                dropout_rate=full_params['dropout_rate'],
                weight_init_std=full_params['weight_init_std'],
                activation=full_params['activation']
            ).to(device)
            
            # 早停机制
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # 自定义训练循环，支持早停
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=full_params['learning_rate'], 
                weight_decay=full_params['weight_decay']
            )
            
            if full_params['use_lr_scheduler']:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=full_params['lr_decay_steps'], 
                    gamma=full_params['lr_decay_rate']
                )
            
            # 数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(train_data['x']),
                torch.FloatTensor(train_data['y']),
                torch.FloatTensor(train_data['t'])
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=full_params['batch_size'], 
                shuffle=True
            )
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                for batch_x, batch_y, batch_t in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_t = batch_t.to(device)
                    
                    if len(batch_t.shape) == 1:
                        batch_t = batch_t.unsqueeze(1)
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(1)
                    
                    optimizer.zero_grad()
                    _, predictions = model(batch_x, batch_t)
                    loss = torch.nn.functional.mse_loss(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                if full_params['use_lr_scheduler']:
                    scheduler.step()
                
                # 验证阶段
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(val_data['x']).to(device)
                    y_val_tensor = torch.FloatTensor(val_data['y']).to(device)
                    t_val_tensor = torch.FloatTensor(val_data['t']).to(device)
                    
                    if len(t_val_tensor.shape) == 1:
                        t_val_tensor = t_val_tensor.unsqueeze(1)
                    if len(y_val_tensor.shape) == 1:
                        y_val_tensor = y_val_tensor.unsqueeze(1)
                    
                    _, predictions = model(X_val_tensor, t_val_tensor)
                    val_loss = torch.nn.functional.mse_loss(predictions, y_val_tensor).item()
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 早停触发
                if patience_counter >= train_patience:
                    break
            
            # 恢复最佳模型状态
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            return best_val_loss
            
        except Exception as e:
            print(f"训练出错: {e}")
            return float('inf')
    
    # 对每个参数组合进行多次运行
    for i, param_combination in enumerate(param_combinations):
        params_dict = dict(zip(param_names, param_combination))
        
        print(f"进度: {i+1}/{len(param_combinations)} - 参数: {params_dict}")
        
        # 运行多次取平均值
        losses = []
        for run in range(n_runs):
            set_seed(run + 50)
            loss = train_single_run(params_dict)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        print(f"  平均损失: {avg_loss:.6f} ± {std_loss:.6f}")
        
        # 更新最佳参数
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_params = params_dict
            no_improvement_count = 0  # 重置计数器
            print(f"  *** 新的最佳参数! 平均损失: {best_avg_loss:.6f} ***")
        else:
            no_improvement_count += 1
            print(f"  连续 {no_improvement_count} 轮无改善")
        
        # 优化早停检查
        if no_improvement_count >= search_patience:
            print(f"  *** 优化早停触发! 连续 {search_patience} 轮无改善，提前停止搜索 ***")
            break
    
    print(f"\nCFR精简网格搜索完成!")
    print(f"搜索了 {i+1}/{len(param_combinations)} 个参数组合")
    print(f"最优参数: {best_params}")
    print(f"最优平均损失: {best_avg_loss:.6f}")
    
    # 返回最优参数（包含固定参数）
    final_best_params = {
        'batch_size': 500,
        'rep_layers': 3,
        'hyp_layers': 3,
        'dropout_rate': 0.1,
        'weight_init_std': 0.1,
        'learning_rate': 5e-4,       # 降低学习率
        'weight_decay': 1e-4,
        'lr_decay_rate': 0.97,       # 衰减更慢
        'lr_decay_steps': 200,       # 衰减间隔拉大
        'use_lr_scheduler': True,
        'split_output': True,
        'varsel': False,
        'batch_norm': True,
        'activation': 'relu',
        'imb_type': 'mmd_lin',
        'loss_type': 'l2',
        'reweight_sample': False,
        'use_p_correction': False,
        **best_params
    }
    
    return final_best_params


def optimize_tar(X_train: np.ndarray, 
                 X_val: np.ndarray, 
                 y_train: np.ndarray, 
                 y_val: np.ndarray, 
                 T_train: np.ndarray, 
                 T_val: np.ndarray,
                 n_runs: int = 5,
                 epochs: int = 100,
                 train_patience: int = 10,
                 search_patience: int = 20) -> Dict[str, Any]:
    """
    TARNet模型精简网格搜索调参并返回最优参数
    
    推荐配置：只搜索最重要的参数
    - 高重要性参数（搜索）：rep_hidden_dim, hyp_hidden_dim, batch_size
    - 中等重要性参数（固定）：rep_layers, hyp_layers
    - 低重要性参数（固定）：dropout_rate, learning_rate, weight_decay等
    
    Args:
        n_runs: 每个参数组合运行的次数，用于取平均值
        patience: 早停耐心值，连续多少轮验证损失没有改善就停止
    
    Returns:
        best_params: 最优参数字典
    """
    import itertools
    
    # 精简参数搜索空间 - 重点搜learning_rate和网络宽度
    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3, 5e-3],  # 4个值，这是收敛的关键
        'rep_hidden_dim': [100, 200],                  # 2个值，不搜50（太小）
        'hyp_hidden_dim': [100, 200],                  # 2个值
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"TARNet精简网格搜索开始 - 总参数组合数: {len(param_combinations)}")
    print(f"每个组合运行次数: {n_runs}")
    print(f"搜索参数: {list(param_grid.keys())}")
    print(f"搜索早停耐心值: {search_patience}")
    
    # 对 y 做标准化
    y_train_raw = y_train.values if hasattr(y_train, 'values') else y_train
    y_val_raw = y_val.values if hasattr(y_val, 'values') else y_val
    
    y_mean = float(np.mean(y_train_raw))
    y_std_val = float(np.std(y_train_raw))
    if y_std_val == 0:
        y_std_val = 1.0
    
    
    
    # 准备数据（使用标准化后的 y）
    train_data = {
        'x': X_train.astype(np.float32),
        'y': ((y_train_raw - y_mean) / y_std_val).astype(np.float32),
        't': T_train.values if hasattr(T_train, 'values') else T_train.astype(np.float32)
    }
    
    val_data = {
        'x': X_val.astype(np.float32),
        'y': ((y_val_raw - y_mean) / y_std_val).astype(np.float32),
        't': T_val.values if hasattr(T_val, 'values') else T_val.astype(np.float32)
    }
    
    input_dim = X_train.shape[1]
    
    # 记录最佳参数
    best_avg_loss = float('inf')
    best_params = None
    no_improvement_count = 0  # 优化早停计数器
    
    def train_single_run(params_dict):
        """单次训练运行"""
        try:
            # 固定参数（batch_size固定200，learning_rate由搜索决定）
            fixed_params = {
                'rep_layers': 3,
                'hyp_layers': 3,
                'batch_size': 200,           # 固定batch_size，不搜了
                'dropout_rate': 0.1,
                'weight_init_std': 0.1,
                'weight_decay': 1e-4,
                'lr_decay_rate': 0.97,
                'lr_decay_steps': 200,
                'use_lr_scheduler': True,
                'use_batch_p': False,
                'varsel': False,
                'batch_norm': True,
                'activation': 'relu',
                'loss_type': 'l2',
                'reweight_sample': False,
                'use_p_correction': False
            }
            
            # 合并参数
            full_params = {**fixed_params, **params_dict}
            
            model = TARNet(
                input_dim=input_dim,
                rep_hidden_dim=full_params['rep_hidden_dim'],
                hyp_hidden_dim=full_params['hyp_hidden_dim'],
                rep_layers=full_params['rep_layers'],
                hyp_layers=full_params['hyp_layers'],
                varsel=full_params['varsel'],
                batch_norm=full_params['batch_norm'],
                dropout_rate=full_params['dropout_rate'],
                weight_init_std=full_params['weight_init_std'],
                activation=full_params['activation']
            ).to(device)
            
            # 早停机制
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # 自定义训练循环，支持早停
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=full_params['learning_rate'], 
                weight_decay=full_params['weight_decay']
            )
            
            if full_params['use_lr_scheduler']:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=full_params['lr_decay_steps'], 
                    gamma=full_params['lr_decay_rate']
                )
            
            # 数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(train_data['x']),
                torch.FloatTensor(train_data['y']),
                torch.FloatTensor(train_data['t'])
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=full_params['batch_size'], 
                shuffle=True
            )
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0.0
                for batch_x, batch_y, batch_t in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    batch_t = batch_t.to(device)
                    
                    if len(batch_t.shape) == 1:
                        batch_t = batch_t.unsqueeze(1)
                    if len(batch_y.shape) == 1:
                        batch_y = batch_y.unsqueeze(1)
                    
                    optimizer.zero_grad()
                    _, predictions = model(batch_x, batch_t)
                    loss = torch.nn.functional.mse_loss(predictions, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                if full_params['use_lr_scheduler']:
                    scheduler.step()
                
                # 验证阶段
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(val_data['x']).to(device)
                    y_val_tensor = torch.FloatTensor(val_data['y']).to(device)
                    t_val_tensor = torch.FloatTensor(val_data['t']).to(device)
                    
                    if len(t_val_tensor.shape) == 1:
                        t_val_tensor = t_val_tensor.unsqueeze(1)
                    if len(y_val_tensor.shape) == 1:
                        y_val_tensor = y_val_tensor.unsqueeze(1)
                    
                    _, predictions = model(X_val_tensor, t_val_tensor)
                    val_loss = torch.nn.functional.mse_loss(predictions, y_val_tensor).item()
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 早停触发
                if patience_counter >= train_patience:
                    break
            
            # 恢复最佳模型状态
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            return best_val_loss
            
        except Exception as e:
            print(f"训练出错: {e}")
            return float('inf')
    
    # 对每个参数组合进行多次运行
    for i, param_combination in enumerate(param_combinations):
        params_dict = dict(zip(param_names, param_combination))
        
        print(f"进度: {i+1}/{len(param_combinations)} - 参数: {params_dict}")
        
        # 运行多次取平均值
        losses = []
        for run in range(n_runs):
            set_seed(run + 100)
            loss = train_single_run(params_dict)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        print(f"  平均损失: {avg_loss:.6f} ± {std_loss:.6f}")
        
        # 更新最佳参数
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_params = params_dict
            no_improvement_count = 0
            print(f"  *** 新的最佳参数! 平均损失: {best_avg_loss:.6f} ***")
        else:
            no_improvement_count += 1
            print(f"  连续 {no_improvement_count} 轮无改善")
        
        # 优化早停检查
        if no_improvement_count >= search_patience:
            print(f"  *** 优化早停触发! 连续 {search_patience} 轮无改善，提前停止搜索 ***")
            break
    
    print(f"\nTARNet精简网格搜索完成!")
    print(f"搜索了 {i+1}/{len(param_combinations)} 个参数组合")
    print(f"最优参数: {best_params}")
    print(f"最优平均损失: {best_avg_loss:.6f}")
    
    # 返回最优参数（包含固定参数 + 搜索到的最优参数）
    final_best_params = {
        'rep_layers': 3,
        'hyp_layers': 3,
        'batch_size': 200,
        'dropout_rate': 0.1,
        'weight_init_std': 0.1,
        'weight_decay': 1e-4,
        'lr_decay_rate': 0.97,
        'lr_decay_steps': 200,
        'use_lr_scheduler': True,
        'use_batch_p': False,
        'varsel': False,
        'batch_norm': True,
        'activation': 'relu',
        'loss_type': 'l2',
        'reweight_sample': False,
        'use_p_correction': False,
        **best_params  # 包含搜索到的 learning_rate, rep_hidden_dim, hyp_hidden_dim
    }
    
    return final_best_params


def optimize_dragonnet(X_train: np.ndarray, 
                       X_val: np.ndarray, 
                       y_train: np.ndarray, 
                       y_val: np.ndarray, 
                       T_train: np.ndarray, 
                       T_val: np.ndarray,
                       n_runs: int = 5,
                       epochs: int = 100,
                       train_patience: int = 10,
                       search_patience: int = 20) -> Dict[str, Any]:
    """
    DragonNet模型精简网格搜索调参并返回最优参数
    
    推荐配置：只搜索最重要的参数
    - 高重要性参数（搜索）：alpha, beta, shared_hidden, outcome_hidden, batch_size
    - 低重要性参数（固定）：learning_rate, loss_type, data_loader_num_workers
    
    Args:
        n_runs: 每个参数组合运行的次数，用于取平均值
        patience: 早停耐心值，连续多少轮验证损失没有改善就停止
    
    Returns:
        best_params: 最优参数字典
    """
    import itertools
    
    # 针对 Bank 数据集优化的搜索空间
    param_grid = {
        'learning_rate': [5e-5, 1e-4],
        'shared_hidden': [64, 100],
        'outcome_hidden': [32, 50],
        'alpha': [0.1, 0.5],
        'beta': [0.1, 0.5],
        'batch_size': [500],
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"DragonNet精简网格搜索开始 - 总参数组合数: {len(param_combinations)}")
    print(f"每个组合运行次数: {n_runs}")
    print(f"搜索参数: {list(param_grid.keys())}")
    print(f"搜索早停耐心值: {search_patience}")
    
    # 对 y 做标准化
    y_train_raw = y_train.values if hasattr(y_train, 'values') else y_train
    y_val_raw = y_val.values if hasattr(y_val, 'values') else y_val
    
    y_mean = float(np.mean(y_train_raw))
    y_std_val = float(np.std(y_train_raw))
    if y_std_val == 0:
        y_std_val = 1.0
    
    
    
   
    train_data = {
        'x': X_train.astype(np.float32),
        'y': ((y_train_raw - y_mean) / y_std_val).astype(np.float32),
        't': T_train.values if hasattr(T_train, 'values') else T_train.astype(np.float32)
    }
    
    val_data = {
        'x': X_val.astype(np.float32),
        'y': ((y_val_raw - y_mean) / y_std_val).astype(np.float32),
        't': T_val.values if hasattr(T_val, 'values') else T_val.astype(np.float32)
    }
    
    input_dim = X_train.shape[1]
    
    # 记录最佳参数
    best_avg_loss = float('inf')
    best_params = None
    no_improvement_count = 0  # 优化早停计数器
    
    def train_single_run(params_dict):
        """单次训练运行"""
        try:
            # 固定参数 + 搜索参数
            fixed_params = {
                'loss_type': 'tarreg',
                'data_loader_num_workers': 0
            }
            
            # 合并参数
            full_params = {**fixed_params, **params_dict}
            
            # 创建DragonNet模型
            model = DragonNet(
                input_dim=input_dim,
                shared_hidden=full_params['shared_hidden'],
                outcome_hidden=full_params['outcome_hidden'],
                alpha=full_params['alpha'],
                beta=full_params['beta'],
                epochs=epochs,
                batch_size=full_params['batch_size'],
                learning_rate=full_params['learning_rate'],
                data_loader_num_workers=full_params['data_loader_num_workers'],
                loss_type=full_params['loss_type'],
                device=device
            )
            
            # 早停机制
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            # 自定义训练循环，支持早停
            optimizer = torch.optim.Adam(
                model.model.parameters(), 
                lr=full_params['learning_rate']
            )
            
            # 数据加载器
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(train_data['x']),
                torch.FloatTensor(train_data['y']),
                torch.FloatTensor(train_data['t'])
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=full_params['batch_size'], 
                shuffle=True
            )
            
            for epoch in range(epochs):
                # 训练阶段
                model.model.train()
                train_loss = 0.0
                for batch_x, batch_y, batch_t in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device).reshape(-1, 1)
                    batch_t = batch_t.to(device).reshape(-1, 1)
                    
                    optimizer.zero_grad()
                    y0_pred, y1_pred, t_pred, eps = model.model(batch_x)
                    
                    # DragonNet损失计算
                    loss = model.loss_f(batch_y, batch_t, t_pred, y0_pred, y1_pred, eps)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # 验证阶段
                model.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    X_val_tensor = torch.FloatTensor(val_data['x']).to(device)
                    y_val_tensor = torch.FloatTensor(val_data['y']).reshape(-1, 1).to(device)
                    t_val_tensor = torch.FloatTensor(val_data['t']).reshape(-1, 1).to(device)
                    
                    y0_pred, y1_pred, t_pred, eps = model.model(X_val_tensor)
                    actual_pred = t_val_tensor * y1_pred + (1 - t_val_tensor) * y0_pred
                    val_loss = torch.nn.functional.mse_loss(actual_pred, y_val_tensor).item()
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # 早停触发
                if patience_counter >= train_patience:
                    break
            
            # 恢复最佳模型状态
            if best_model_state is not None:
                model.model.load_state_dict(best_model_state)
            
            return best_val_loss
            
        except Exception as e:
            print(f"训练出错: {e}")
            return float('inf')
    
    # 对每个参数组合进行多次运行
    for i, param_combination in enumerate(param_combinations):
        params_dict = dict(zip(param_names, param_combination))
        
        print(f"进度: {i+1}/{len(param_combinations)} - 参数: {params_dict}")
        
        # 运行多次取平均值
        losses = []
        for run in range(n_runs):
            set_seed(run + 200)
            loss = train_single_run(params_dict)
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        print(f"  平均损失: {avg_loss:.6f} ± {std_loss:.6f}")
        
        # 更新最佳参数
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_params = params_dict
            no_improvement_count = 0  # 重置计数器
            print(f"  *** 新的最佳参数! 平均损失: {best_avg_loss:.6f} ***")
        else:
            no_improvement_count += 1
            print(f"  连续 {no_improvement_count} 轮无改善")
        
        # 优化早停检查
        if no_improvement_count >= search_patience:
            print(f"  *** 优化早停触发! 连续 {search_patience} 轮无改善，提前停止搜索 ***")
            break
    
    print(f"\nDragonNet精简网格搜索完成!")
    print(f"搜索了 {i+1}/{len(param_combinations)} 个参数组合")
    print(f"最优参数: {best_params}")
    print(f"最优平均损失: {best_avg_loss:.6f}")
    
    # 返回最优参数（包含固定参数 + 搜索到的最优参数）
    final_best_params = {
        'loss_type': 'tarreg',
        'data_loader_num_workers': 0,
        **best_params  # 包含搜索到的 learning_rate, shared_hidden, outcome_hidden, alpha, beta, batch_size
    }
    
    return final_best_params


def train_cfr(X_train: np.ndarray, 
                              X_val: np.ndarray, 
                              y_train: np.ndarray, 
                              y_val: np.ndarray, 
                              T_train: np.ndarray, 
                              T_val: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              T_test: np.ndarray,
                              best_params: Dict[str, Any],
                              epochs: int = 100,
                              train_patience: int = 10) -> pd.DataFrame:
    """
    使用最优参数训练CFR模型并预测CATE
    
    Args:
        X_train, X_val, y_train, y_val, T_train, T_val: 训练和验证数据
        X_test, y_test, T_test: 测试数据集用于最终预测
        best_params: 最优参数字典
        epochs: 训练轮数
        patience: 早停耐心值
    
    Returns:
        DataFrame: 包含模型名称和CATE预测值的DataFrame
    """
    # 准备训练数据
    train_data = {
        'x': X_train.astype(np.float32),
        'y': y_train.values if hasattr(y_train, 'values') else y_train.astype(np.float32),
        't': T_train.values if hasattr(T_train, 'values') else T_train.astype(np.float32)
    }
    
    val_data = {
        'x': X_val.astype(np.float32),
        'y': y_val.values if hasattr(y_val, 'values') else y_val.astype(np.float32),
        't': T_val.values if hasattr(T_val, 'values') else T_val.astype(np.float32)
    }
    
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = CFRNet(
        input_dim=input_dim,
        rep_hidden_dim=best_params['rep_hidden_dim'],
        hyp_hidden_dim=best_params['hyp_hidden_dim'],
        rep_layers=best_params['rep_layers'],
        hyp_layers=best_params['hyp_layers'],
        split_output=best_params['split_output'],
        varsel=best_params['varsel'],
        batch_norm=best_params['batch_norm'],
        dropout_rate=best_params['dropout_rate'],
        weight_init_std=best_params['weight_init_std'],
        activation=best_params['activation']
    ).to(device)
    
    # 训练模型
    train_CFR(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        alpha=best_params['alpha'],
        lr_decay_rate=best_params['lr_decay_rate'],
        lr_decay_steps=best_params['lr_decay_steps'],
        use_lr_scheduler=best_params['use_lr_scheduler'],
        imb_type=best_params['imb_type'],
        use_p_correction=best_params['use_p_correction'],
        reweight_sample=best_params['reweight_sample'],
        loss_type=best_params['loss_type'],
        output_delay=epochs + 1,
        patience=train_patience,
        device=device
    )
    
    # 预测CATE
    model.eval()
    with torch.no_grad():
        X_test = np.array(X_test)
        T_test = np.array(T_test)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        T_test_tensor = torch.FloatTensor(T_test).to(device)
        
        if len(T_test_tensor.shape) == 1:
            T_test_tensor = T_test_tensor.unsqueeze(1)
        
        # 预测处理组和对照组的潜在结果
        _, y0_pred = model(X_test_tensor, torch.zeros_like(T_test_tensor))
        _, y1_pred = model(X_test_tensor, torch.ones_like(T_test_tensor))
        
        # 计算CATE
        cate_pred = y1_pred - y0_pred
        
        # 转换为numpy数组
        cate_pred = cate_pred.cpu().numpy().flatten()
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'CFR': cate_pred
    })
    
    return result_df


def train_tar(X_train: np.ndarray, 
                              X_val: np.ndarray, 
                              y_train: np.ndarray, 
                              y_val: np.ndarray, 
                              T_train: np.ndarray, 
                              T_val: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              T_test: np.ndarray,
                              best_params: Dict[str, Any],
                              epochs: int = 100,
                              train_patience: int = 10) -> pd.DataFrame:
    """
    使用最优参数训练TARNet模型并预测CATE
    
    Args:
        X_train, X_val, y_train, y_val, T_train, T_val: 训练和验证数据
        X_full, y_full, T_full: 完整数据集用于最终预测
        best_params: 最优参数字典
        epochs: 训练轮数
    
    Returns:
        DataFrame: 包含模型名称和CATE预测值的DataFrame
    """
    
    y_train_raw = y_train.values if hasattr(y_train, 'values') else y_train
    y_val_raw = y_val.values if hasattr(y_val, 'values') else y_val
    
    y_mean = float(np.mean(y_train_raw))
    y_std_val = float(np.std(y_train_raw))
    if y_std_val == 0:
        y_std_val = 1.0
    
    y_train_scaled = ((y_train_raw - y_mean) / y_std_val).astype(np.float32)
    y_val_scaled = ((y_val_raw - y_mean) / y_std_val).astype(np.float32)
    
    
    
    
    train_data = {
        'x': X_train.astype(np.float32),
        'y': y_train_scaled,
        't': T_train.values if hasattr(T_train, 'values') else T_train.astype(np.float32)
    }
    
    val_data = {
        'x': X_val.astype(np.float32),
        'y': y_val_scaled,
        't': T_val.values if hasattr(T_val, 'values') else T_val.astype(np.float32)
    }
    
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = TARNet(
        input_dim=input_dim,
        rep_hidden_dim=best_params['rep_hidden_dim'],
        hyp_hidden_dim=best_params['hyp_hidden_dim'],
        rep_layers=best_params['rep_layers'],
        hyp_layers=best_params['hyp_layers'],
        varsel=best_params['varsel'],
        batch_norm=best_params['batch_norm'],
        dropout_rate=best_params['dropout_rate'],
        weight_init_std=best_params['weight_init_std'],
        activation=best_params['activation']
    ).to(device)
    
    # Bank 数据集启用梯度裁剪
    model.use_grad_clip = True
    
    # 训练模型
    training_history = train_TAR(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=epochs,
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        lr_decay_rate=best_params['lr_decay_rate'],
        lr_decay_steps=best_params['lr_decay_steps'],
        use_lr_scheduler=best_params['use_lr_scheduler'],
        use_batch_p=best_params['use_batch_p'],
        use_p_correction=best_params['use_p_correction'],
        reweight_sample=best_params['reweight_sample'],
        loss_type=best_params['loss_type'],
        output_delay=1,
        patience=train_patience,
        device=device
    )
    
    # 记录训练信息
    epochs_run = len(training_history.get('total_loss', []))
    best_val = min(training_history.get('val_total_loss', [float('inf')])) if training_history.get('val_total_loss') else None
    final_train_loss = training_history.get('total_loss', [None])[-1] if training_history.get('total_loss') else None
    final_val_loss = training_history.get('val_total_loss', [None])[-1] if training_history.get('val_total_loss') else None
    print(f"[TARNet] 训练完成: 实际epochs={epochs_run}, best_val_loss={best_val:.4f}, final_train_loss={final_train_loss:.4f}, final_val_loss={final_val_loss:.4f}" if best_val else f"[TARNet] 训练完成: 实际epochs={epochs_run}")
    # 预测CATE
    model.eval()
    with torch.no_grad():
        X_test = np.array(X_test)
        T_test = np.array(T_test)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        T_test_tensor = torch.FloatTensor(T_test.values if hasattr(T_test, 'values') else T_test).to(device)
        
        if len(T_test_tensor.shape) == 1:
            T_test_tensor = T_test_tensor.unsqueeze(1)
        
        # 预测处理组和对照组的潜在结果
        _, y0_pred = model(X_test_tensor, torch.zeros_like(T_test_tensor))
        _, y1_pred = model(X_test_tensor, torch.ones_like(T_test_tensor))
        
        
        cate_pred = (y1_pred - y0_pred) * y_std_val
        
        # 转换为numpy数组
        cate_pred = cate_pred.cpu().numpy().flatten()
    
    # 打印 CATE 统计信息（log）
    print(f"[TARNet] CATE预测统计: mean={np.mean(cate_pred):.4f}, std={np.std(cate_pred):.4f}, min={np.min(cate_pred):.4f}, max={np.max(cate_pred):.4f}")
    
    # 记录到 training_log.csv
    log_training_result(
        model_name='TARNet',
        params=best_params,
        epochs_run=epochs_run,
        best_val_loss=best_val,
        cate_mean=np.mean(cate_pred),
        cate_std=np.std(cate_pred),
        cate_min=np.min(cate_pred),
        cate_max=np.max(cate_pred),
        note=f"y_std={y_std_val:.2f}, train_loss={final_train_loss:.4f}, val_loss={final_val_loss:.4f}" if final_train_loss else f"y_std={y_std_val:.2f}"
    )
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({
        'TAR': cate_pred
    })
    
    return result_df


def train_dragonnet(X_train: np.ndarray, 
                                    X_val: np.ndarray, 
                                    y_train: np.ndarray, 
                                    y_val: np.ndarray, 
                                    T_train: np.ndarray, 
                                    T_val: np.ndarray,
                                    X_test: np.ndarray,
                                    y_test: np.ndarray,
                                    T_test: np.ndarray,
                                    best_params: Dict[str, Any],
                                    epochs: int = 100,
                                    train_patience: int = 10) -> pd.DataFrame:
    """
    使用最优参数训练DragonNet模型并预测CATE
    
    Args:
        X_train, X_val, y_train, y_val, T_train, T_val: 训练和验证数据
        X_full, y_full, T_full: 完整数据集用于最终预测
        best_params: 最优参数字典
        epochs: 训练轮数
    
    Returns:
        DataFrame: 包含模型名称和CATE预测值的DataFrame
    """
    # 对 y 做标准化
    y_train_raw = y_train.values if hasattr(y_train, 'values') else y_train
    
    y_mean = float(np.mean(y_train_raw))
    y_std_val = float(np.std(y_train_raw))
    if y_std_val == 0:
        y_std_val = 1.0
    
    y_train_scaled = ((y_train_raw - y_mean) / y_std_val).astype(np.float32)
    
    print(f"[DragonNet] y标准化: mean={y_mean:.4f}, std={y_std_val:.4f}")
    
    
    train_data = {
        'x': X_train.astype(np.float32),
        'y': y_train_scaled,
        't': T_train.values if hasattr(T_train, 'values') else T_train.astype(np.float32)
    }
    
    input_dim = X_train.shape[1]
    
    # 创建模型
    model = DragonNet(
        input_dim=input_dim,
        shared_hidden=best_params['shared_hidden'],
        outcome_hidden=best_params['outcome_hidden'],
        alpha=best_params['alpha'],
        beta=best_params['beta'],
        epochs=epochs,
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        data_loader_num_workers=0,  # 固定为0避免CUDA多进程错误
        loss_type=best_params['loss_type'],
        device=device
    )
    
    # Bank 数据集启用梯度裁剪
    model.use_grad_clip = True
    
    # 训练模型
    train_info = model.fit(
        x=train_data['x'],
        y=train_data['y'],
        t=train_data['t'],
        valid_perc=0.2,
        patience=train_patience
    )
    
    print(f"[DragonNet] 训练完成: 实际epochs={train_info['epochs_run']}, best_val_loss={train_info['best_val_loss']:.4f}" if train_info['best_val_loss'] else f"[DragonNet] 训练完成: 实际epochs={train_info['epochs_run']}")
    
    # 预测CATE
    model.model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        # DragonNet直接输出y0和y1的预测
        y0_pred, y1_pred, t_pred, eps = model.model(X_test_tensor)
        
        # 计算CATE（反标准化：乘回 y_std）
        cate_pred = (y1_pred - y0_pred) * y_std_val
        
        # 转换为numpy数组
        cate_pred = cate_pred.cpu().numpy().flatten()
    
    # 打印 CATE 统计信息（log）
    print(f"[DragonNet] CATE预测统计: mean={np.mean(cate_pred):.4f}, std={np.std(cate_pred):.4f}, min={np.min(cate_pred):.4f}, max={np.max(cate_pred):.4f}")
    
    # 记录到 training_log.csv
    log_training_result(
        model_name='DragonNet',
        params=best_params,
        epochs_run=train_info['epochs_run'],
        best_val_loss=train_info['best_val_loss'],
        cate_mean=np.mean(cate_pred),
        cate_std=np.std(cate_pred),
        cate_min=np.min(cate_pred),
        cate_max=np.max(cate_pred),
        note=f"y_std={y_std_val:.2f}, final_train_loss={train_info['final_train_loss']:.4f}" if train_info['final_train_loss'] else f"y_std={y_std_val:.2f}"
    )
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({        
        'Dragon': cate_pred
    })
    
    return result_df


