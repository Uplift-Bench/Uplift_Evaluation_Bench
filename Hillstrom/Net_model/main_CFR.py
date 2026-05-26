from .base_net import *
from .util import mmd2_lin, mmd2_rbf, wasserstein
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class CFRLoss(nn.Module):
    """
    CFR (Counterfactual Regression) 损失函数
    
    对应TensorFlow训练代码中的损失组件：
    - tot_loss: 总损失 (优化目标)
    - pred_loss: 预测损失 (事实结果误差)
    - imb_dist: 不平衡距离损失 (分布平衡)
    - cf_error: 反事实误差 (仅评估用)
    - weight_decay: 权重衰减损失
    """
    
    def __init__(self, 
                 loss_type: str = 'l2',           # 对应 FLAGS.loss
                 alpha: float = 1e-4,             # 对应 FLAGS.p_alpha (不平衡正则化)
                 lambda_reg: float = 0.0,         # 对应 FLAGS.p_lambda (权重衰减)
                 imb_type: str = 'mmd_lin',       # 对应 FLAGS.imb_fun
                 use_p_correction: bool = True,   # 对应 FLAGS.use_p_correction
                 reweight_sample: bool = True):   # 对应 FLAGS.reweight_sample
        """
        Args:
            loss_type: 预测损失类型 ('l1', 'l2', 'log')
            alpha: 不平衡正则化权重
            lambda_reg: 权重衰减正则化权重
            imb_type: 不平衡惩罚函数类型
            use_p_correction: 是否使用处理概率校正
            reweight_sample: 是否重新加权样本
        """
        super(CFRLoss, self).__init__()
        
        self.loss_type = loss_type
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.imb_type = imb_type
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
        计算预测损失 (对应 CFR.pred_loss)
        
        Args:
            y_pred: 预测值 (batch_size, 1)
            y_true: 真实值 (batch_size, 1)
            t: 处理变量 (batch_size, 1)
            p_t: 处理概率
            
        Returns:
            预测损失值
        """
        if self.reweight_sample:
            
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
    
    def compute_imbalance_loss(self, 
                             representations: torch.Tensor,
                             t: torch.Tensor,
                             p_t: float) -> torch.Tensor:
        """
        计算不平衡距离损失 (对应 CFR.imb_dist)
        
        Args:
            representations: 表示层输出 (batch_size, dim)
            t: 处理变量 (batch_size, 1)
            p_t: 处理概率
            
        Returns:
            不平衡距离损失值
        """
        t_flat = t.flatten()
        
        # 分离处理组和对照组的表示
        treated_mask = t_flat > 0
        control_mask = t_flat < 1
        
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            return torch.tensor(0.0, device=representations.device)
        
        rep_treated = representations[treated_mask]
        rep_control = representations[control_mask]
        
        if self.imb_type == 'mmd_lin':
            # 线性MMD (Maximum Mean Discrepancy)
            imb_loss = self._compute_mmd_linear(rep_treated, rep_control, p_t)
        elif self.imb_type == 'mmd_rbf':
            # RBF MMD
            imb_loss = self._compute_mmd_rbf(rep_treated, rep_control, p_t)
        elif self.imb_type == 'wass':
            # Wasserstein distance
            imb_loss = self._compute_wasserstein(rep_treated, rep_control, p_t)
        else:
            raise ValueError(f"Unsupported imbalance type: {self.imb_type}")
        
        return imb_loss
    
    def _compute_mmd_linear(self, 
                          rep_treated: torch.Tensor,
                          rep_control: torch.Tensor,
                          p_t: float) -> torch.Tensor:
        """计算线性MMD距离，使用util.py中的mmd2_lin函数"""
        # 重构数据：合并处理组和对照组
        rep_combined = torch.cat([rep_control, rep_treated], dim=0)
        
        # 创建处理变量向量：对照组为0，处理组为1
        t_combined = torch.cat([
            torch.zeros(rep_control.shape[0], 1, device=rep_control.device),
            torch.ones(rep_treated.shape[0], 1, device=rep_treated.device)
        ], dim=0)
        
        # 使用util.py中的mmd2_lin函数
        mmd = mmd2_lin(rep_combined, t_combined.flatten(), p_t)
        
        return mmd
    
    def _compute_mmd_rbf(self, 
                        rep_treated: torch.Tensor,
                        rep_control: torch.Tensor,
                        p_t: float,
                        sigma: float = 0.1) -> torch.Tensor:
        """计算RBF MMD距离，使用util.py中的mmd2_rbf函数"""
        # 重构数据：合并处理组和对照组
        rep_combined = torch.cat([rep_control, rep_treated], dim=0)
        
        # 创建处理变量向量：对照组为0，处理组为1
        t_combined = torch.cat([
            torch.zeros(rep_control.shape[0], 1, device=rep_control.device),
            torch.ones(rep_treated.shape[0], 1, device=rep_treated.device)
        ], dim=0)
        
        # 使用util.py中的mmd2_rbf函数
        mmd = mmd2_rbf(rep_combined, t_combined.flatten(), p_t, sigma)
        
        return mmd
    
    def _compute_wasserstein(self, 
                           rep_treated: torch.Tensor,
                           rep_control: torch.Tensor,
                           p_t: float,
                           lam: float = 1.0,
                           its: int = 10) -> torch.Tensor:
        """计算Wasserstein距离，使用util.py中的wasserstein函数"""
        # 重构数据：合并处理组和对照组
        rep_combined = torch.cat([rep_control, rep_treated], dim=0)
        
        # 创建处理变量向量：对照组为0，处理组为1
        t_combined = torch.cat([
            torch.zeros(rep_control.shape[0], 1, device=rep_control.device),
            torch.ones(rep_treated.shape[0], 1, device=rep_treated.device)
        ], dim=0)
        
        # 使用util.py中的wasserstein函数
        wass_dist, _ = wasserstein(rep_combined, t_combined.flatten(), p_t, lam, its)
        
        return wass_dist
    
    
    
    def compute_weight_decay_loss(self, model: nn.Module) -> torch.Tensor:
        """
        计算权重衰减损失
        
        Args:
            model: 神经网络模型
            
        Returns:
            权重衰减损失值
        """
        if self.lambda_reg <= 0 or model is None:
            return torch.tensor(0.0, device=next(model.parameters()).device if model else 'cpu')
        
        weight_decay_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight_decay_loss += torch.norm(param, p=2) ** 2
        
        return self.lambda_reg * weight_decay_loss
    
    def forward(self, 
                y_pred: torch.Tensor,
                y_true: torch.Tensor,
                representations: torch.Tensor,
                t: torch.Tensor,
                p_t: float,
                model: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """
        计算完整的CFR损失
        
        Args:
            y_pred: 预测值 (batch_size, 1)
            y_true: 真实值 (batch_size, 1)
            representations: 表示层输出 (batch_size, dim)
            t: 处理变量 (batch_size, 1)
            p_t: 处理概率
            model: 神经网络模型（用于权重衰减）
            
        Returns:
            包含各损失组件的字典
        """
        # 1. 预测损失 (对应 CFR.pred_loss)
        pred_loss = self.compute_prediction_loss(y_pred, y_true, t, p_t)
        
        # 2. 不平衡距离损失 (对应 CFR.imb_dist)
        imb_loss = self.compute_imbalance_loss(representations, t, p_t)
        
        # 3. 权重衰减损失
        weight_decay_loss = self.compute_weight_decay_loss(model)
        
        # 4. 总损失 (对应 CFR.tot_loss)
        total_loss = pred_loss + self.alpha * imb_loss + weight_decay_loss
        
        return {
            'total_loss': total_loss,      # 对应 CFR.tot_loss
            'pred_loss': pred_loss,        # 对应 CFR.pred_loss  
            'imb_loss': imb_loss,          # 对应 CFR.imb_dist
            'weight_decay_loss': weight_decay_loss,
            'f_error': pred_loss,          # 事实误差别名
            'imb_err': imb_loss           # 不平衡误差别名
        }
    
    def compute_counterfactual_error(self,
                                   y_pred_cf: torch.Tensor,
                                   y_true_cf: torch.Tensor,
                                   t_cf: torch.Tensor,
                                   p_t: float) -> torch.Tensor:
        """
        计算反事实误差 (对应 cf_error)
        
        Args:
            y_pred_cf: 反事实预测值
            y_true_cf: 反事实真实值
            t_cf: 反转的处理变量
            p_t: 处理概率
            
        Returns:
            反事实误差值
        """
        return self.compute_prediction_loss(y_pred_cf, y_true_cf, t_cf, p_t)

class CFRNet(nn.Module):
    """
    完整的CFR模型，结合表示网络和假设网络
    """
    def __init__(self, 
                 input_dim: int,
                 rep_hidden_dim: int = 200,
                 hyp_hidden_dim: int = 100,
                 rep_layers: int = 3,
                 hyp_layers: int = 2,
                 split_output: bool = False,
                 varsel: bool = False,
                 batch_norm: bool = False,
                 dropout_rate: float = 0.1,
                 weight_init_std: float = 0.1,
                 activation: str = 'relu'):
        super(CFRNet, self).__init__()
        
        # 表示网络
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
        
        # 假设网络
        rep_output_dim = self.representation_net.get_representation_dim()
        self.hypothesis_net = HypothesisNet(
            input_dim=rep_output_dim,
            hidden_dim=hyp_hidden_dim,
            output_dim=hyp_hidden_dim,
            n_layers=hyp_layers,
            split_output=split_output,
            dropout_rate=dropout_rate,
            weight_init_std=weight_init_std,
            activation=activation
        )
        
        self.split_output = split_output
    
    def forward(self, x, t):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch_size, input_dim)
            t: 处理变量 (batch_size, 1)
            
        Returns:
            representations: 表示层输出
            predictions: 预测结果
        """
        # 表示网络
        representations = self.representation_net(x)
        
        # 假设网络
        predictions = self.hypothesis_net(representations, t)
        
        return representations, predictions
    
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
    
  


def train_CFR(model: CFRNet, 
              train_data: Dict, 
              val_data: Dict = None,
              epochs: int = 1000,
              batch_size: int = 100,
              learning_rate: float = 0.05,  
              weight_decay: float = 0.0,
              alpha: float = 1e-4,
              # 新增学习率调度参数
              lr_decay_rate: float = 0.95,  
              lr_decay_steps: int = 100,    
              use_lr_scheduler: bool = True,  
              imb_type: str = 'mmd_lin',
              use_p_correction: bool = True,
              reweight_sample: bool = True,
              loss_type: str = 'l2',
              output_delay: int = 100,
              patience: int = 15,
              device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    
    
    Args:
        model: CFR模型
        train_data: 训练数据字典，包含 'x', 't', 'y' 键
        val_data: 验证数据字典（可选）
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 初始学习率 
        weight_decay: 权重衰减
        alpha: 不平衡正则化权重
        lr_decay_rate: 学习率衰减率 
        lr_decay_steps: 学习率衰减步数 
        use_lr_scheduler: 是否启用学习率调度 
        imb_type: 不平衡惩罚类型
        use_p_correction: 是否使用处理概率校正
        reweight_sample: 是否重新加权样本
        loss_type: 损失函数类型
        output_delay: 输出间隔
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
    
    # 计算处理概率
    p_treated = float(t_train.mean())
    print(f"Treatment probability: {p_treated:.4f}")
    
    # 模型移动到设备
    model = model.to(device)
    
    # 创建损失函数
    criterion = CFRLoss(
        loss_type=loss_type,
        alpha=alpha,
        lambda_reg=weight_decay,
        imb_type=imb_type,
        use_p_correction=use_p_correction,
        reweight_sample=reweight_sample
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    scheduler = None
    if use_lr_scheduler:
        # 计算每个epoch需要多少步
        n_train = X_train.shape[0]
        steps_per_epoch = (n_train + batch_size - 1) // batch_size
        
       
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=lr_decay_steps,  
            gamma=lr_decay_rate        
        )
        print(f"Learning rate scheduler enabled: decay rate={lr_decay_rate}, decay steps={lr_decay_steps}")
    
    # 创建早停器
    early_stopper = EarlyStopper(patience=patience)
    
    # 训练历史记录
    training_history = {
        'total_loss': [],
        'pred_loss': [],
        'imb_loss': [],
        'weight_decay_loss': [],
        'val_total_loss': [],
        'val_pred_loss': [],
        'val_imb_loss': [],
        'cf_error': [],
        'learning_rates': []  # 新增：记录学习率变化
    }
    
    n_train = X_train.shape[0]
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Training samples: {n_train}")
    print(f"Batch size: {batch_size}")
    print(f"Initial learning rate: {learning_rate}")
    print(f"Alpha (imbalance weight): {alpha}")
    print(f"Imbalance type: {imb_type}")
    print("-" * 50)
    
    
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
            
            # 计算损失
            loss_dict = criterion(
                y_pred=predictions,
                y_true=y_batch,
                representations=representations,
                t=t_batch,
                p_t=p_treated,
                model=model
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            optimizer.step()
            
            
            if scheduler is not None:
                scheduler.step()
            
            global_step += 1
            
            epoch_losses.append({
                'total_loss': loss_dict['total_loss'].item(),
                'pred_loss': loss_dict['pred_loss'].item(),
                'imb_loss': loss_dict['imb_loss'].item(),
                'weight_decay_loss': loss_dict['weight_decay_loss'].item()
            })
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        training_history['learning_rates'].append(current_lr)
        
        # 记录训练损失
        avg_train_loss = {
            'total_loss': np.mean([l['total_loss'] for l in epoch_losses]),
            'pred_loss': np.mean([l['pred_loss'] for l in epoch_losses]),
            'imb_loss': np.mean([l['imb_loss'] for l in epoch_losses]),
            'weight_decay_loss': np.mean([l['weight_decay_loss'] for l in epoch_losses])
        }
        
        training_history['total_loss'].append(avg_train_loss['total_loss'])
        training_history['pred_loss'].append(avg_train_loss['pred_loss'])
        training_history['imb_loss'].append(avg_train_loss['imb_loss'])
        training_history['weight_decay_loss'].append(avg_train_loss['weight_decay_loss'])
        
        # 验证阶段
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_representations, val_predictions = model(X_val, t_val)
                val_loss_dict = criterion(
                    y_pred=val_predictions,
                    y_true=y_val,
                    representations=val_representations,
                    t=t_val,
                    p_t=p_treated,
                    model=model
                )
                
                training_history['val_total_loss'].append(val_loss_dict['total_loss'].item())
                training_history['val_pred_loss'].append(val_loss_dict['pred_loss'].item())
                training_history['val_imb_loss'].append(val_loss_dict['imb_loss'].item())
                
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
                        val_cf_pred, y_cf_val, t_cf_val, p_treated
                    )
                    training_history['cf_error'].append(cf_error.item())
        
        # 输出训练进度
        if epoch % output_delay == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d}/{epochs} [Step {global_step:5d}]")
            print(f"  Train - Total: {avg_train_loss['total_loss']:.4f}, "
                  f"Pred: {avg_train_loss['pred_loss']:.4f}, "
                  f"Imb: {avg_train_loss['imb_loss']:.2e}, "
                  f"WD: {avg_train_loss['weight_decay_loss']:.2e}")
            print(f"  Learning Rate: {current_lr:.6f}")  # 显示当前学习率
            
            if val_data is not None:
                print(f"  Val   - Total: {training_history['val_total_loss'][-1]:.4f}, "
                      f"Pred: {training_history['val_pred_loss'][-1]:.4f}, "
                      f"Imb: {training_history['val_imb_loss'][-1]:.2e}")
                
                if 'y_cf' in val_data:
                    print(f"  CF Error: {training_history['cf_error'][-1]:.4f}")
            
            print("-" * 50)
    
    print("Training completed!")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    return training_history




def create_cfr_model(input_dim: int, config: Dict = None) -> CFRNet:
    """
    创建CFR模型的便捷函数
    
    Args:
        input_dim: 输入特征维度
        config: 模型配置
    
    Returns:
        CFR模型实例
    """
    if config is None:
        config = {}
    
    return CFRNet(
        input_dim=input_dim,
        rep_hidden_dim=config.get('rep_hidden_dim', 200),
        hyp_hidden_dim=config.get('hyp_hidden_dim', 100),
        rep_layers=config.get('rep_layers', 3),
        hyp_layers=config.get('hyp_layers', 2),
        split_output=config.get('split_output', False),
        varsel=config.get('varsel', False),
        batch_norm=config.get('batch_norm', False),
        dropout_rate=config.get('dropout_rate', 0.1),
        weight_init_std=config.get('weight_init_std', 0.1),
        activation=config.get('activation', 'relu')
    )











