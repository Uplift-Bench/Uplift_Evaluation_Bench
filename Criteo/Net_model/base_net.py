from .util import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


class RepresentationNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, 
                 varsel=False, batch_norm=False, dropout_rate=0.0, 
                 weight_init_std=0.1, activation='relu'):
        """
        表示网络组件
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            n_layers: 隐藏层数量
            varsel: 是否使用变量选择模式
            batch_norm: 是否使用批量归一化
            dropout_rate: dropout概率
            weight_init_std: 权重初始化标准差
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid', 'elu')
        """
        super(RepresentationNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.varsel = varsel
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.weight_init_std = weight_init_std
        
        # 选择激活函数
        self.activation = self._get_activation(activation)
        
        # 构建网络层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        for i in range(n_layers):
            if i == 0:
                # 第一层处理
                if varsel:
                    # 变量选择模式：为每个输入特征分配一个权重（标量）
                    self.feature_weights = nn.Parameter(
                        torch.ones(input_dim) / input_dim
                    )
                    # 变量选择模式下第一层不需要线性变换
                    self.layers.append(None)
                else:
                    # 标准模式：输入层到第一隐藏层
                    layer = nn.Linear(input_dim, hidden_dim)
                    self._init_weights(layer, input_dim)
                    self.layers.append(layer)
            else:
                # 后续隐藏层
                if self.varsel and i == 1:
                    # 变量选择模式下第二层的输入维度是input_dim
                    layer = nn.Linear(input_dim, hidden_dim)
                    self._init_weights(layer, input_dim)
                else:
                    layer = nn.Linear(hidden_dim, hidden_dim)
                    self._init_weights(layer, hidden_dim)
                self.layers.append(layer)
            
            # 批量归一化层
            if batch_norm:
                if i == 0 and varsel:
                    # 变量选择模式下第一层后的BN维度是input_dim
                    self.batch_norms.append(nn.BatchNorm1d(input_dim))
                else:
                    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def _get_activation(self, activation):
        """获取激活函数"""
        if activation == 'relu':
            return F.relu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'elu':
            return F.elu
        else:
            return F.relu
    
    def _init_weights(self, layer, input_dim):
        """初始化权重"""
        std = self.weight_init_std / np.sqrt(input_dim)
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
            
        Returns:
            表示层输出，形状为 (batch_size, hidden_dim)
        """
        h = x  # 初始化输入
        
        for i in range(self.n_layers):
            if i == 0 and self.varsel:
                # 变量选择模式：第一层直接进行特征加权
                h = h * self.feature_weights  # 逐元素相乘实现特征加权
            else:
                # 标准线性变换
                if self.layers[i] is not None:
                    h = self.layers[i](h)
            
            # 批量归一化
            if self.batch_norm and self.batch_norms is not None:
                h = self.batch_norms[i](h)
            
            # 激活函数（除了变量选择模式的第一层）
            if not (i == 0 and self.varsel):
                h = self.activation(h)
            
            # Dropout
            if self.dropout is not None:
                h = self.dropout(h)
        
        return h
    
    def get_representation_dim(self):
        """获取表示层输出维度"""
        if self.varsel and self.n_layers == 1:
            return self.input_dim
        else:
            return self.hidden_dim


class HypothesisNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers,
                 split_output=False, dropout_rate=0.0, weight_init_std=0.1,
                 activation='relu', weight_decay=1.0):
        """
        假设网络组件
        
        Args:
            input_dim: 输入特征维度（表示层输出维度）
            hidden_dim: 隐藏层维度
            output_dim: 输出层维度
            n_layers: 输出网络层数
            split_output: 是否为处理组和对照组分别构建独立网络
            dropout_rate: dropout概率
            weight_init_std: 权重初始化标准差
            activation: 激活函数类型
            weight_decay: 权重衰减系数
        """
        super(HypothesisNet, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.split_output = split_output
        self.dropout_rate = dropout_rate
        self.weight_init_std = weight_init_std
        self.weight_decay = weight_decay
        
        # 选择激活函数
        self.activation = self._get_activation(activation)
        
        if split_output:
            # 为处理组和对照组分别构建独立的假设网络
            self.hypothesis_net_0 = self._build_single_output_net(input_dim)  # 对照组
            self.hypothesis_net_1 = self._build_single_output_net(input_dim)  # 处理组
        else:
            # 将处理变量作为额外输入特征的单一网络
            self.hypothesis_net = self._build_single_output_net(input_dim + 1)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def _get_activation(self, activation):
        """获取激活函数"""
        if activation == 'relu':
            return F.relu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'elu':
            return F.elu
        else:
            return F.relu
    
    def _build_single_output_net(self, input_dim):
        """构建单个输出网络"""
        layers = nn.ModuleList()
        
        # 构建输出层网络
        dims = [input_dim] + [self.output_dim] * self.n_layers
        
        for i in range(self.n_layers):
            layer = nn.Linear(dims[i], dims[i+1])
            self._init_weights(layer, dims[i])
            layers.append(layer)
        
        # 最终预测层
        pred_layer = nn.Linear(self.output_dim, 1)
        self._init_weights(pred_layer, self.output_dim)
        layers.append(pred_layer)
        
        return layers
    
    def _init_weights(self, layer, input_dim):
        """初始化权重"""
        std = self.weight_init_std / np.sqrt(input_dim)
        nn.init.normal_(layer.weight, mean=0.0, std=std)
        nn.init.zeros_(layer.bias)
    
    def _forward_single_net(self, x, net_layers):
        """单个网络的前向传播"""
        h = x
        
        # 通过输出层网络
        for i in range(self.n_layers):
            h = net_layers[i](h)
            h = self.activation(h)
            if self.dropout is not None:
                h = self.dropout(h)
        
        # 最终预测层
        y = net_layers[-1](h)  # 最后一层是预测层
        
        return y
    
    def forward(self, rep, t):
        """
        前向传播
        
        Args:
            rep: 表示层输出，形状为 (batch_size, input_dim)
            t: 处理变量，形状为 (batch_size, 1)
            
        Returns:
            y: 预测输出，形状为 (batch_size, 1)
        """
        if self.split_output:
            # 分别为处理组和对照组构建独立的假设网络
            
            # 找到处理组和对照组的索引
            t_flat = t.flatten()
            i0 = torch.where(t_flat < 1)[0]  # 对照组索引
            i1 = torch.where(t_flat > 0)[0]  # 处理组索引
            
            # 分别获取对照组和处理组的表示
            rep0 = rep[i0]
            rep1 = rep[i1]
            
            # 分别通过对应的网络
            y0 = self._forward_single_net(rep0, self.hypothesis_net_0)
            y1 = self._forward_single_net(rep1, self.hypothesis_net_1)
            
            # 合并结果
            y = torch.zeros(rep.shape[0], 1, device=rep.device)
            y[i0] = y0
            y[i1] = y1
            
        else:
            # 将处理变量作为额外输入特征
            h_input = torch.cat([rep, t], dim=1)
            y = self._forward_single_net(h_input, self.hypothesis_net)
        
        return y
    
    def get_weights_for_regularization(self):
        """获取用于正则化的权重"""
        weights = []
        
        if self.split_output:
            # 收集两个网络的权重
            for layer in self.hypothesis_net_0:
                if hasattr(layer, 'weight'):
                    weights.append(layer.weight)
            for layer in self.hypothesis_net_1:
                if hasattr(layer, 'weight'):
                    weights.append(layer.weight)
        else:
            # 收集单个网络的权重
            for layer in self.hypothesis_net:
                if hasattr(layer, 'weight'):
                    weights.append(layer.weight)
        
        return weights
    
    def compute_weight_decay_loss(self):
        """计算权重衰减损失"""
        if self.weight_decay <= 0:
            return 0.0
        
        weights = self.get_weights_for_regularization()
        weight_decay_loss = 0.0
        
        for weight in weights:
            weight_decay_loss += torch.norm(weight, p=2) ** 2
        
        return self.weight_decay * weight_decay_loss


