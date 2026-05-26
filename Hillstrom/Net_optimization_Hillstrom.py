

import torch
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
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
        'alpha': [10**(k/2) for k in range(-8, 5)],  # 13个值：10^-4 到 10^2.5
        'rep_hidden_dim': [50, 100, 200],  # 3个值
        'hyp_hidden_dim': [50, 100, 200],  # 3个值
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
            # 固定参数（基于经验值）
            fixed_params = {
                'batch_size': 500,           # 固定批次大小
                'rep_layers': 3,             # 固定表示层数量为3
                'hyp_layers': 3,             # 固定假设层数量为3
                'dropout_rate': 0.1,         # 固定dropout率
                'weight_init_std': 0.1,      # 固定权重初始化
                'learning_rate': 1e-3,       # 固定学习率
                'weight_decay': 1e-4,        # 固定权重衰减
                'lr_decay_rate': 0.9,        # 固定学习率衰减
                'lr_decay_steps': 100,       # 固定学习率衰减步数
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
        'rep_layers': 3,             # 固定表示层数量为3
        'hyp_layers': 3,             # 固定假设层数量为3
        'dropout_rate': 0.1,
        'weight_init_std': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'lr_decay_rate': 0.9,
        'lr_decay_steps': 100,
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
    
    
    param_grid = {
        'rep_hidden_dim': [50, 100, 200],  # 3个值
        'hyp_hidden_dim': [50, 100, 200],  # 3个值
        'batch_size': [200, 500],  # 2个值
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"TARNet网格搜索开始 - 总参数组合数: {len(param_combinations)}")
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
    
    def train_single_run(params_dict):
        """单次训练运行"""
        try:
            # 固定参数（基于经验值）
            fixed_params = {
                'rep_layers': 3,             # 固定表示层数量为3
                'hyp_layers': 3,             # 固定假设层数量为3
                'dropout_rate': 0.1,         # 固定dropout率
                'weight_init_std': 0.1,      # 固定权重初始化
                'learning_rate': 1e-3,       # 固定学习率
                'weight_decay': 1e-4,        # 固定权重衰减
                'lr_decay_rate': 0.9,        # 固定学习率衰减
                'lr_decay_steps': 100,       # 固定学习率衰减步数
                'use_lr_scheduler': True,    # 固定使用学习率调度器
                'use_batch_p': False,        # 固定批次处理
                'varsel': False,             # 固定变量选择
                'batch_norm': True,          # 固定批量归一化
                'activation': 'relu',        # 固定激活函数
                'loss_type': 'l2',           # 固定损失类型
                'reweight_sample': False,    # 固定样本重加权
                'use_p_correction': False    # 固定处理概率校正
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
    
    print(f"\nTARNet精简网格搜索完成!")
    print(f"搜索了 {i+1}/{len(param_combinations)} 个参数组合")
    print(f"最优参数: {best_params}")
    print(f"最优平均损失: {best_avg_loss:.6f}")
    
    # 返回最优参数（包含固定参数）
    final_best_params = {
        'rep_layers': 3,             # 固定表示层数量为3
        'hyp_layers': 3,             # 固定假设层数量为3
        'dropout_rate': 0.1,
        'weight_init_std': 0.1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'lr_decay_rate': 0.9,
        'lr_decay_steps': 100,
        'use_lr_scheduler': True,
        'use_batch_p': False,
        'varsel': False,
        'batch_norm': True,
        'activation': 'relu',
        'loss_type': 'l2',
        'reweight_sample': False,
        'use_p_correction': False,
        **best_params
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
    
    # 精简参数搜索空间 - 只搜索最重要的参数
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0],  # 4个值
        'beta': [0.1, 0.5, 1.0, 2.0],   # 4个值
        'shared_hidden': [100, 200],     # 2个值
        'outcome_hidden': [100, 200],    # 2个值
        'batch_size': [200, 500],        # 2个值
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    print(f"DragonNet精简网格搜索开始 - 总参数组合数: {len(param_combinations)}")
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
            # 固定参数（基于经验值）
            fixed_params = {
                'learning_rate': 1e-3,        # 固定学习率
                'loss_type': 'tarreg',         # 固定损失类型
                'data_loader_num_workers': 0   # 固定数据加载器工作进程数
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
    
    # 返回最优参数（包含固定参数）
    final_best_params = {
        'learning_rate': 1e-3,
        'loss_type': 'tarreg',
        'data_loader_num_workers': 0,
        **best_params
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
    
    # 训练模型
    train_TAR(
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
        T_test_tensor = torch.FloatTensor(T_test.values if hasattr(T_test, 'values') else T_test).to(device)
        
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
    # 准备训练数据
    train_data = {
        'x': X_train.astype(np.float32),
        'y': y_train.values if hasattr(y_train, 'values') else y_train.astype(np.float32),
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
    
    # 训练模型
    model.fit(
        x=train_data['x'],
        y=train_data['y'],
        t=train_data['t'],
        valid_perc=0.2,
        patience=train_patience
    )
    
    # 预测CATE
    model.model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        # DragonNet直接输出y0和y1的预测
        y0_pred, y1_pred, t_pred, eps = model.model(X_test_tensor)
        
        # 计算CATE
        cate_pred = y1_pred - y0_pred
        
        # 转换为numpy数组
        cate_pred = cate_pred.cpu().numpy().flatten()
    
    # 创建结果DataFrame
    result_df = pd.DataFrame({        
        'Dragon': cate_pred
    })
    
    return result_df


def example_usage():
    """
    使用示例：展示如何使用精简网格搜索进行超参数优化
    
    这个示例展示了如何：
    1. 使用精简网格搜索进行超参数优化（只搜索最重要的参数）
    2. 对每个参数组合运行多次取平均值
    3. 使用最优参数训练最终模型
    
    推荐配置：
    - CFR: 搜索alpha, rep_hidden_dim, hyp_hidden_dim (13×3×3=117个组合)
    - TARNet: 搜索rep_hidden_dim, hyp_hidden_dim, batch_size (3×3×2=18个组合)
    - DragonNet: 搜索alpha, beta, shared_hidden, outcome_hidden, batch_size (4×4×2×2×2=128个组合)
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    T = np.random.binomial(1, 0.5, n_samples)
    y = np.random.randn(n_samples) + 0.5 * T + 0.1 * X[:, 0]
    
    # 划分数据
    X_train, X_temp, y_train, y_temp, T_train, T_temp = train_test_split(
        X, y, T, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test, T_val, T_test = train_test_split(
        X_temp, y_temp, T_temp, test_size=0.5, random_state=42
    )
    
    print("=== 超参数优化示例（精简网格搜索）===")
    
    # 1. CFR模型优化（精简网格搜索）
    print("\n1. 优化CFR模型...")
    cfr_params = optimize_cfr(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        T_train=T_train, T_val=T_val,
        n_runs=5, # 每个组合运行5次取平均
        epochs=100,
        patience=5,    # 早停耐心值
    )
    print(f"CFR最优参数: {cfr_params}")
    
    # 2. TARNet模型优化（精简网格搜索）
    print("\n2. 优化TARNet模型...")
    tar_params = optimize_tar(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        T_train=T_train, T_val=T_val,
        n_runs=3,  # 每个参数组合运行3次取平均值
        epochs=100,
        patience=5,    # 早停耐心值
    )
    print(f"TARNet最优参数: {tar_params}")
    
    # 3. DragonNet模型优化（精简网格搜索）
    print("\n3. 优化DragonNet模型...")
    dragon_params = optimize_dragonnet(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        T_train=T_train, T_val=T_val,
        n_runs=3,  # 每个参数组合运行3次取平均值
        epochs=100,
        patience=5,    # 早停耐心值
    )
    print(f"DragonNet最优参数: {dragon_params}")
    
    # 4. 使用最优参数训练最终模型
    print("\n4. 训练最终模型...")
    
    # CFR最终训练
    cfr_results = train_cfr(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        T_train=T_train, T_val=T_val,
        X_test=X_test, y_test=y_test, T_test=T_test,
        best_params=cfr_params,
        epochs=100,
        train_patience=10,
    )
    
    # TARNet最终训练
    tar_results = train_tar(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        T_train=T_train, T_val=T_val,
        X_test=X_test, y_test=y_test, T_test=T_test,
        best_params=tar_params,
        epochs=100,
        train_patience=10,
    )
    
    # DragonNet最终训练
    dragon_results = train_dragonnet(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        T_train=T_train, T_val=T_val,
        X_test=X_test, y_test=y_test, T_test=T_test,
        best_params=dragon_params,
        epochs=100,
        train_patience=10,
    )
    
    # 合并结果
    final_results = pd.concat([cfr_results, tar_results, dragon_results], axis=1)
    print(f"\n最终CATE预测结果形状: {final_results.shape}")
    print(f"各模型CATE预测均值:")
    print(final_results.mean())
    
    return final_results


if __name__ == "__main__":
    # 运行示例
    results = example_usage()