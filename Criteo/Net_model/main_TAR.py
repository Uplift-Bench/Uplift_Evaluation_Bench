from .base_net import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TARLoss(nn.Module):
    """
    TARNet (Treatment-Agnostic Representation Network) 损失函数
    
    与CFRNet的主要区别：
    - 没有不平衡正则化项 (α = 0)
    - 只包含预测损失和权重衰减损失
    
    对应的损失组件：
    - pred_loss: 预测损失 (事实结果误差)
    - weight_decay: 权重衰减损失
    - tot_loss: 总损失 = pred_loss + weight_decay
    """
    
    def __init__(self, 
                 loss_type: str = 'l2',           # 预测损失类型
                 lambda_reg: float = 0.0,         # 权重衰减正则化权重
                 use_p_correction: bool = True,   # 是否使用处理概率校正
                 reweight_sample: bool = True):   # 是否重新加权样本
        """
        Args:
            loss_type: 预测损失类型 ('l1', 'l2', 'log')
            lambda_reg: 权重衰减正则化权重
            use_p_correction: 是否使用处理概率校正
            reweight_sample: 是否重新加权样本
        """
        super(TARLoss, self).__init__()
        
        self.loss_type = loss_type
        self.lambda_reg = lambda_reg
        self.use_p_correction = use_p_correction
        self.reweight_sample = reweight_sample
        
        # 预测损失函数
        if loss_type == 'l1':
            self.pred_loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.pred_loss_fn = F.mse_loss
        elif loss_type == 'log':
            self.pred_loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_prediction_loss(self, 
                              y_pred: torch.Tensor, 
                              y_true: torch.Tensor,
                              t: torch.Tensor,
                              p_t: float) -> torch.Tensor:
        """
        计算预测损失 (与CFRNet相同的实现)
        
        Args:
            y_pred: 预测值 (batch_size, 1)
            y_true: 真实值 (batch_size, 1)
            t: 处理变量 (batch_size, 1)
            p_t: 处理概率
            
        Returns:
            预测损失值
        """
        if self.reweight_sample:
            # 重新加权样本以平衡处理组和对照组
            weights = t / p_t + (1 - t) / (1 - p_t)
            weights = weights / weights.mean()  # 归一化权重
            
            if self.loss_type == 'log':
                # 对数损失需要特殊处理
                loss = F.binary_cross_entropy_with_logits(
                    y_pred, y_true, weight=weights, reduction='mean'
                )
            else:
                # L1/L2损失
                pointwise_loss = self.pred_loss_fn(y_pred, y_true, reduction='none')
                loss = (pointwise_loss * weights).mean()
        else:
            loss = self.pred_loss_fn(y_pred, y_true, reduction='mean')
        
        return loss
    
    def compute_weight_decay_loss(self, model: nn.Module) -> torch.Tensor:
        """
        计算权重衰减损失
        
        Args:
            model: 神经网络模型
            
        Returns:
            权重衰减损失值
        """
        # 推荐使用优化器的weight_decay参数，这里返回0
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    def forward(self, 
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                t: torch.Tensor,
                p_t: float,
                model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        计算完整的TARNet损失
        
        Args:
            y_pred: 预测值 (batch_size, 1)
            y_true: 真实值 (batch_size, 1)
            t: 处理变量 (batch_size, 1)
            p_t: 处理概率
            model: 神经网络模型（用于权重衰减）
            
        Returns:
            包含各损失组件的字典
        """
        # 1. 预测损失
        pred_loss = self.compute_prediction_loss(y_pred, y_true, t, p_t)
        
        # 2. 权重衰减损失
        weight_decay_loss = self.compute_weight_decay_loss(model) if model else 0.0
        
        # 3. 总损失 (TARNet: 没有不平衡正则化项)
        total_loss = pred_loss + weight_decay_loss
        
        return {
            'total_loss': total_loss,
            'pred_loss': pred_loss,
            'weight_decay_loss': weight_decay_loss,
            'f_error': pred_loss,  # 事实误差别名
            'imb_loss': torch.tensor(0.0, device=y_pred.device),  # TARNet中为0
            'imb_err': torch.tensor(0.0, device=y_pred.device)    # TARNet中为0
        }
    
    def compute_counterfactual_error(self,
                                   y_pred_cf: torch.Tensor,
                                   y_true_cf: torch.Tensor,
                                   t_cf: torch.Tensor,
                                   p_t: float) -> torch.Tensor:
        """
        计算反事实误差
        
        Args:
            y_pred_cf: 反事实预测值
            y_true_cf: 反事实真实值
            t_cf: 反转的处理变量
            p_t: 处理概率
            
        Returns:
            反事实误差值
        """
        return self.compute_prediction_loss(y_pred_cf, y_true_cf, t_cf, p_t)


class TARNet(nn.Module):
    """
    TARNet (Treatment-Agnostic Representation Network) 模型
    
    核心特点：
    1. 始终使用split_output=True（TARNet的核心设计）
    2. 为每个处理创建完全独立的假设网络
    3. 支持多处理（不限于二元处理）
    4. 训练时不使用不平衡正则化
    """
    def __init__(self, 
                 input_dim: int,
                 n_treatments: int = 2,              # 处理数量（TARNet特有）
                 rep_hidden_dim: int = 200,
                 hyp_hidden_dim: int = 100,
                 rep_layers: int = 3,
                 hyp_layers: int = 2,
                 varsel: bool = False,
                 batch_norm: bool = False,
                 dropout_rate: float = 0.1,
                 weight_init_std: float = 0.1,
                 activation: str = 'relu',
                 normalizer: Optional[nn.Module] = None):  # 输入归一化
        super(TARNet, self).__init__()
        
        # TARNet的核心特点：始终使用独立的假设网络
        self.n_treatments = n_treatments
        self.normalizer = normalizer
        
        # 表示网络（共享的φ网络）
        self.representation_net = RepresentationNet(
            input_dim=input_dim,
            hidden_dim=rep_hidden_dim,
            n_layers=rep_layers,
            varsel=varsel,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate,
            weight_init_std=weight_init_std,
            activation=activation
        )
        
        # 为每个处理创建独立的假设网络（TARNet核心架构）
        rep_output_dim = self.representation_net.get_representation_dim()
        self.hypothesis_nets = nn.ModuleList([
            HypothesisNet(
                input_dim=rep_output_dim,
                hidden_dim=hyp_hidden_dim,
                output_dim=hyp_hidden_dim,
                n_layers=hyp_layers,
                split_output=False,  # 每个网络内部不需要split
                dropout_rate=dropout_rate,
                weight_init_std=weight_init_std,
                activation=activation
            ) for _ in range(n_treatments)
        ])
        
        # 为了兼容性，保留split_output属性
        self.split_output = True
    
    def forward(self, x, t):
        """
        前向传播（TARNet专用架构）
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            t: 处理变量 (batch_size, 1) 或 (batch_size,)
            
        Returns:
            representations: 表示层输出
            predictions: 预测结果
        """
        # 输入归一化
        if self.normalizer is not None:
            x = self.normalizer(x)
        
        # 表示网络（共享的φ网络）
        representations = self.representation_net(x)
        
        # 确保处理变量的形状正确
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        # TARNet特有：为每个处理使用独立的假设网络
        batch_size = x.shape[0]
        predictions = torch.zeros(batch_size, 1, device=x.device)
        
        # 为每个处理分别计算预测
        for treatment_idx in range(self.n_treatments):
            # 找到对应处理的样本
            treatment_mask = (t.squeeze() == treatment_idx)
            
            if treatment_mask.any():
                # 获取该处理的样本表示
                treatment_representations = representations[treatment_mask]
                
                # 使用对应的假设网络（不需要传入t，因为网络已经是处理特定的）
                treatment_predictions = self.hypothesis_nets[treatment_idx](
                    treatment_representations, 
                    torch.zeros(treatment_representations.shape[0], 1, device=x.device)
                )
                
                # 将预测结果放回原位置
                predictions[treatment_mask] = treatment_predictions
        
        return representations, predictions
    
    def predict_counterfactual(self, x, t):
        """
        预测反事实结果（TARNet专用）
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            t: 处理变量 (batch_size, 1) 或 (batch_size,)
            
        Returns:
            factual_pred: 事实预测
            counterfactual_pred: 反事实预测
        """
        # 输入归一化
        if self.normalizer is not None:
            x = self.normalizer(x)
        
        # 获取表示
        representations = self.representation_net(x)
        
        # 确保处理变量的形状正确
        if len(t.shape) == 1:
            t = t.unsqueeze(1)
        
        batch_size = x.shape[0]
        factual_pred = torch.zeros(batch_size, 1, device=x.device)
        counterfactual_pred = torch.zeros(batch_size, 1, device=x.device)
        
        # 为每个样本预测事实和反事实结果
        for i in range(batch_size):
            sample_rep = representations[i:i+1]
            sample_t = int(t[i].item())
            
            # 事实预测：使用实际处理的网络
            factual_pred[i] = self.hypothesis_nets[sample_t](
                sample_rep, 
                torch.zeros(1, 1, device=x.device)
            )
            
            # 反事实预测：使用相反处理的网络
            if self.n_treatments == 2:
                # 二元处理：简单反转
                cf_t = 1 - sample_t
                counterfactual_pred[i] = self.hypothesis_nets[cf_t](
                    sample_rep, 
                    torch.zeros(1, 1, device=x.device)
                )
            else:
                # 多处理：预测其他所有处理的平均值
                cf_preds = []
                for cf_t in range(self.n_treatments):
                    if cf_t != sample_t:
                        cf_pred = self.hypothesis_nets[cf_t](
                            sample_rep, 
                            torch.zeros(1, 1, device=x.device)
                        )
                        cf_preds.append(cf_pred)
                counterfactual_pred[i] = torch.stack(cf_preds).mean(dim=0)
        
        return factual_pred, counterfactual_pred
    
    def predict(self, x, t):
        """
        仅进行预测，只返回预测结果
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            t: 处理变量 (batch_size, 1)
            
        Returns:
            predictions: 预测结果 (batch_size, 1)
        """
        representations, predictions = self.forward(x, t)
        return predictions
    
   


def train_TAR(model: TARNet, 
              train_data: Dict, 
              val_data: Dict = None,
              epochs: int = 1000,
              batch_size: int = 100,
              learning_rate: float = 0.05,  # 修改为与TensorFlow相同的默认值
              weight_decay: float = 0.0,
              # 新增学习率调度参数
              lr_decay_rate: float = 0.95,  # 对应TensorFlow的lrate_decay
              lr_decay_steps: int = 100,    # 对应TensorFlow的NUM_ITERATIONS_PER_DECAY
              use_lr_scheduler: bool = True,  # 是否启用学习率调度
              use_p_correction: bool = True,
              reweight_sample: bool = True,
              loss_type: str = 'l2',
              output_delay: int = 100,
              use_batch_p: bool = False,
              patience: int = 15,
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    训练TARNet模型 - 包含与TensorFlow版本相同的学习率调度
    
    Args:
        model: TARNet模型
        train_data: 训练数据字典，包含 'x', 't', 'y' 键
        val_data: 验证数据字典（可选）
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 初始学习率 (对应TensorFlow的FLAGS.lrate)
        weight_decay: 权重衰减
        lr_decay_rate: 学习率衰减率 (对应TensorFlow的FLAGS.lrate_decay)
        lr_decay_steps: 学习率衰减步数 (对应TensorFlow的NUM_ITERATIONS_PER_DECAY)
        use_lr_scheduler: 是否启用学习率调度 (对应TensorFlow的exponential_decay)
        use_p_correction: 是否使用处理概率校正
        reweight_sample: 是否重新加权样本
        loss_type: 损失函数类型
        output_delay: 输出间隔
        use_batch_p: 是否使用批次级处理概率
        device: 设备
    
    Returns:
        training_history: 训练历史记录
    """
    # 数据准备
    X_train = torch.FloatTensor(train_data['x']).to(device)
    t_train = torch.FloatTensor(train_data['t']).to(device)
    y_train = torch.FloatTensor(train_data['y']).to(device)
    
    if val_data is not None:
        X_val = torch.FloatTensor(val_data['x']).to(device)
        t_val = torch.FloatTensor(val_data['t']).to(device)
        y_val = torch.FloatTensor(val_data['y']).to(device)
    
    # 确保处理变量和标签的维度正确
    if len(t_train.shape) == 1:
        t_train = t_train.unsqueeze(1)
    if len(y_train.shape) == 1:
        y_train = y_train.unsqueeze(1)
    
    if val_data is not None:
        if len(t_val.shape) == 1:
            t_val = t_val.unsqueeze(1)
        if len(y_val.shape) == 1:
            y_val = y_val.unsqueeze(1)
    
    # 计算全局处理概率
    p_treated_global = float(t_train.mean())
    print(f"Global treatment probability: {p_treated_global:.4f}")
    if use_batch_p:
        print("Using batch-level treatment probability")
    else:
        print("Using global treatment probability")
    
    # 模型移动到设备
    model = model.to(device)
    
    # 创建损失函数 (TARNet: 没有不平衡正则化)
    criterion = TARLoss(
        loss_type=loss_type,
        lambda_reg=0.0,  # 使用优化器的weight_decay代替
        use_p_correction=use_p_correction,
        reweight_sample=reweight_sample
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 创建学习率调度器 (对应TensorFlow的exponential_decay)
    scheduler = None
    if use_lr_scheduler:
        # 计算每个epoch需要多少步
        n_train = X_train.shape[0]
        steps_per_epoch = (n_train + batch_size - 1) // batch_size
        
        # PyTorch的StepLR实现TensorFlow的exponential_decay with staircase=True
        # 每lr_decay_steps步衰减一次，衰减率为lr_decay_rate
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=lr_decay_steps,  # 对应NUM_ITERATIONS_PER_DECAY
            gamma=lr_decay_rate        # 对应FLAGS.lrate_decay
        )
        print(f"Learning rate scheduler enabled: decay rate={lr_decay_rate}, decay steps={lr_decay_steps}")
    
    # 创建早停器
    early_stopper = EarlyStopper(patience=patience)
    
    # 训练历史记录
    training_history = {
        'total_loss': [],
        'pred_loss': [],
        'weight_decay_loss': [],
        'val_total_loss': [],
        'val_pred_loss': [],
        'cf_error': [],
        'learning_rates': []  # 新增：记录学习率变化
    }
    
    n_train = X_train.shape[0]
    
    print(f"Starting TARNet training for {epochs} epochs...")
    print(f"Training samples: {n_train}")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"TARNet特点: 无不平衡正则化 (α = 0)")
    print("-" * 50)
    
    # 全局步数计数器 (对应TensorFlow的global_step)
    global_step = 0
    
    for epoch in range(epochs):
        model.train()
        
        # 随机打乱训练数据
        indices = torch.randperm(n_train)
        epoch_losses = []
        
        # 批次训练
        for i in range(0, n_train, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            x_batch = X_train[batch_indices]
            t_batch = t_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # 前向传播
            representations, predictions = model(x_batch, t_batch)
            
            # 计算批次级处理概率
            if use_batch_p:
                p_treated_batch = float(t_batch.mean())
                # 如果批次中只有一种类型，使用全局概率
                if p_treated_batch == 0.0 or p_treated_batch == 1.0:
                    p_treated_batch = p_treated_global
            else:
                p_treated_batch = p_treated_global
            
            # 计算损失
            loss_dict = criterion(
                y_pred=predictions,
                y_true=y_batch,
                t=t_batch,
                p_t=p_treated_batch,
                model=model
            )
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss_dict['total_loss'].backward()
            # 梯度裁剪，防止极端样本导致梯度爆炸（仅Bank数据集需要）
            if hasattr(model, 'use_grad_clip') and model.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 更新学习率调度器 (每步更新，对应TensorFlow的global_step)
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            
            epoch_losses.append({
                'total_loss': loss_dict['total_loss'].item(),
                'pred_loss': loss_dict['pred_loss'].item(),
                'weight_decay_loss': loss_dict['weight_decay_loss'].item()
            })
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rates'].append(current_lr)
        
        # 记录训练损失
        avg_train_loss = {
            'total_loss': np.mean([l['total_loss'] for l in epoch_losses]),
            'pred_loss': np.mean([l['pred_loss'] for l in epoch_losses]),
            'weight_decay_loss': np.mean([l['weight_decay_loss'] for l in epoch_losses])
        }
        
        training_history['total_loss'].append(avg_train_loss['total_loss'])
        training_history['pred_loss'].append(avg_train_loss['pred_loss'])
        training_history['weight_decay_loss'].append(avg_train_loss['weight_decay_loss'])
        
        # 验证阶段
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_representations, val_predictions = model(X_val, t_val)
                val_loss_dict = criterion(
                    y_pred=val_predictions,
                    y_true=y_val,
                    t=t_val,
                    p_t=p_treated_global,
                    model=model
                )
                
                training_history['val_total_loss'].append(val_loss_dict['total_loss'].item())
                training_history['val_pred_loss'].append(val_loss_dict['pred_loss'].item())
                
                # 早停检查
                if early_stopper.early_stop(val_loss_dict['total_loss'].item()):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # 计算反事实误差（如果有真实的反事实标签）
                if 'y_cf' in val_data:
                    y_cf_val = torch.FloatTensor(val_data['y_cf']).to(device)
                    if len(y_cf_val.shape) == 1:
                        y_cf_val = y_cf_val.unsqueeze(1)
                    
                    t_cf_val = 1 - t_val
                    val_cf_pred = model.hypothesis_net(val_representations, t_cf_val)
                    cf_error = criterion.compute_counterfactual_error(
                        val_cf_pred, y_cf_val, t_cf_val, p_treated_global
                    )
                    training_history['cf_error'].append(cf_error.item())
        
        # 输出训练进度
        if epoch % output_delay == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs} [Step {global_step:5d}]")
            print(f"  Train - Total: {avg_train_loss['total_loss']:.4f}, "
                  f"Pred: {avg_train_loss['pred_loss']:.4f}, "
                  f"WD: {avg_train_loss['weight_decay_loss']:.2e}")
            print(f"  Learning Rate: {current_lr:.6f}")  # 显示当前学习率
            
            if val_data is not None:
                print(f"  Val   - Total: {training_history['val_total_loss'][-1]:.4f}, "
                      f"Pred: {training_history['val_pred_loss'][-1]:.4f}")
                
                if 'y_cf' in val_data:
                    print(f"  CF Error: {training_history['cf_error'][-1]:.4f}")
            
            print("-" * 50)
    
    print("TARNet training completed!")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    return training_history


def create_tar_model(input_dim: int, config: Dict = None) -> TARNet:
    """
    创建TARNet模型的便捷函数
    
    Args:
        input_dim: 输入特征维度
        config: 模型配置
    
    Returns:
        TARNet模型实例
    """
    if config is None:
        config = {}
    
    return TARNet(
        input_dim=input_dim,
        n_treatments=config.get('n_treatments', 2),
        rep_hidden_dim=config.get('rep_hidden_dim', 200),
        hyp_hidden_dim=config.get('hyp_hidden_dim', 100),
        rep_layers=config.get('rep_layers', 3),
        hyp_layers=config.get('hyp_layers', 2),
        varsel=config.get('varsel', False),
        batch_norm=config.get('batch_norm', False),
        dropout_rate=config.get('dropout_rate', 0.1),
        weight_init_std=config.get('weight_init_std', 0.1),
        activation=config.get('activation', 'relu'),
        normalizer=config.get('normalizer', None)
    )



