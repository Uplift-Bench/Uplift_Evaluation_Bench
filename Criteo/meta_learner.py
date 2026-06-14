import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
import torch
from sklearn.model_selection import train_test_split
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def model_prepare(model_name, X_train, y_train, X_val, y_val, n_trials=50, xgb_opt_early_stopping_rounds=10, trial_patience=20):
    """
    模型准备和Optuna自动调参函数 - 修改版避免数据泄露
    
    Args:
        model_name: 模型名称 ('xgb', 'lr')
        X_train: 训练集特征矩阵
        y_train: 训练集目标变量
        X_val: 验证集特征矩阵（仅用于最终评估，不参与调优）
        y_val: 验证集目标变量（仅用于最终评估，不参与调优）
        n_trials: Optuna优化试验次数
        xgb_opt_early_stopping_rounds: XGBoost早停轮数
        trial_patience: trial级别早停耐心值，连续多少个trial没有改善就停止整个优化
    
    Returns:
        调参后的最优模型和最优参数
    """
    
    def objective_xgb(trial):
        """XGBoost优化目标函数 - 使用训练集内部交叉验证"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
            'tree_method': 'hist',  # 使用hist方法            
            'random_state': 42
        }     
        model = XGBRegressor(**params, device='cuda',early_stopping_rounds=xgb_opt_early_stopping_rounds)
        model.fit(X_train, y_train,eval_set=[(X_val, y_val)],verbose=False)
        val_pred = model.predict(X_val)
        # 优化指标为r2_score方向为最大
        return r2_score(y_val, val_pred) 
    
    def objective_lr(trial):
        """LogisticRegression优化目标函数 - 使用训练集内部交叉验证"""
        params = {
            'C': trial.suggest_float('C', 0.01, 10.0),
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        # 优化指标为r2方向为最大
        return r2_score(y_val, val_pred)
    
    # 设置Optuna日志级别为WARNING以减少输出
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    if model_name == 'xgb':
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        
        # 记录最佳值和trial计数器
        best_value = float('-inf')  
        trial_counter = 0
        
        def objective_with_trial_stopping(trial):
            nonlocal best_value, trial_counter
            
            result = objective_xgb(trial)
            
            # 检查是否有改善
            if result > best_value:
                best_value = result
                trial_counter = 0
            else:
                trial_counter += 1
            
            # trial级别早停检查
            if trial_counter >= trial_patience:
                print(f"Trial-level early stopping: {trial_patience} consecutive trials without improvement")
                raise optuna.TrialPruned("Trial-level early stopping triggered")
            
            return result
        
        study.optimize(objective_with_trial_stopping, n_trials=n_trials)
        best_params_raw = study.best_params
        return best_params_raw
            
    elif model_name == 'lr':
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        
        # 记录最佳值和trial计数器
        best_value = float('-inf')  
        trial_counter = 0
        
        def objective_with_trial_stopping(trial):
            nonlocal best_value, trial_counter
            
            result = objective_lr(trial)
            
            # 检查是否有改善
            if result > best_value:
                best_value = result
                trial_counter = 0
            else:
                trial_counter += 1
            
            # trial级别早停检查
            if trial_counter >= trial_patience:
                print(f"Trial-level early stopping: {trial_patience} consecutive trials without improvement")
                raise optuna.TrialPruned("Trial-level early stopping triggered")
            
            return result
        
        study.optimize(objective_with_trial_stopping, n_trials=n_trials)
        best_params_raw = study.best_params
        
        # 清理参数字典，只保留LogisticRegression需要的参数
        best_params = {
            'C': best_params_raw.get('C'),  # 提供默认值
            'penalty': 'l2',  # 固定为'l2'
            'solver': 'lbfgs',
            'max_iter': 1000,  # 最大迭代次数
            'fit_intercept': True,  # 固定为True
            'random_state': 42
        }
        return best_params
        
    else:
        # 默认使用XGBoost
        print(f"不支持的模型类型: {model_name}，使用默认XGBoost")
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'tree_method': 'hist',  # 使用hist方法
            'device': 'cuda',  # 固定使用GPU加速
            'predictor': 'gpu_predictor',  # 使用GPU进行预测
            'random_state': 42
        }
        return default_params
def prepare_y(X_train, T_train, y_train,X_val,T_val,y_val, n_trials=50, y_opt_early_stopping_rounds=10, trial_patience=20):
    """
    准备mu0和mu1预测值 - 修改版避免数据泄露
    
    Args:
        X_full: 完整特征矩阵
        T_full: 完整处理变量
        y_full: 完整结果变量
        X_train: 训练集特征矩阵
        T_train: 训练集处理变量
        y_train: 训练集结果变量
        X_val: 验证集特征矩阵（仅用于最终评估）
        T_val: 验证集处理变量（仅用于最终评估）
        y_val: 验证集结果变量（仅用于最终评估）
        n_trials: Optuna试验次数
        y_opt_early_stopping_rounds: 超参数调优时的早停轮数
        y_train_early_stopping_rounds: 最终训练时的早停轮数
        trial_patience: trial级别早停耐心值
    """
    treatment_index_train = T_train == 1
    control_index_train = T_train == 0

    treatment_X_train = X_train[treatment_index_train]
    treatment_y_train = y_train[treatment_index_train]
    
    control_X_train = X_train[control_index_train]
    control_y_train = y_train[control_index_train] 

    treatment_index_val = T_val == 1
    control_index_val = T_val == 0
    
    treatment_X_val = X_val[treatment_index_val]
    treatment_y_val = y_val[treatment_index_val]
    
    control_X_val = X_val[control_index_val]
    control_y_val = y_val[control_index_val]

    # 为control组单独调参和训练mu0模型  
    mu0_params = model_prepare(model_name='xgb', X_train=control_X_train, y_train=control_y_train, 
                               X_val=control_X_val, y_val=control_y_val,
                               n_trials=n_trials, xgb_opt_early_stopping_rounds=y_opt_early_stopping_rounds, trial_patience=trial_patience)
    mu0_model = XGBRegressor(**mu0_params, device='cuda')
    
    mu0_model.fit(control_X_train, control_y_train)

    # 为treatment组单独调参和训练mu1模型  
    mu1_params = model_prepare(model_name='xgb', X_train=treatment_X_train, y_train=treatment_y_train, 
                               X_val=treatment_X_val, y_val=treatment_y_val,
                               n_trials=n_trials, xgb_opt_early_stopping_rounds=y_opt_early_stopping_rounds, trial_patience=trial_patience)
    mu1_model = XGBRegressor(**mu1_params, device='cuda')    
  
    mu1_model.fit(treatment_X_train, treatment_y_train)

    mu0_pred = mu0_model.predict(X_train)
    mu1_pred = mu1_model.predict(X_train)

    return mu0_pred, mu1_pred
    
def S_learner(X_train, T_train, y_train,X_test,X_val, y_val, n_trials_xgb=50, xgb_opt_early_stopping_rounds=10, trial_patience=20):
    """
    S-learner实现 - 修改版避免数据泄露
    
    Args:
        X_full: 完整特征矩阵
        T_full: 完整处理变量
        y_full: 完整结果变量
        X_train: 训练集特征矩阵
        T_train: 训练集处理变量
        y_train: 训练集结果变量
        X_val: 验证集特征矩阵（仅用于最终评估）
        T_val: 验证集处理变量（仅用于最终评估）
        y_val: 验证集结果变量（仅用于最终评估）
        n_trials: Optuna试验次数
        xgb_opt_early_stopping_rounds: 超参数调优时的早停轮数
        xgb_train_early_stopping_rounds: 最终训练时的早停轮数
        trial_patience: trial级别早停耐心值
    """
    # 获取最优参数 - 使用训练集内部交叉验证
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                                n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    
    # 训练S-learner模型
    # 将pandas Series转换为numpy数组
    T_train_array = T_train.values if hasattr(T_train, 'values') else np.array(T_train)
    tx = np.concatenate([X_train, T_train_array.reshape(-1, 1)], axis=1)
    s_model = XGBRegressor(**xgb_params, device='cuda')
    
    # 训练S-learner模型，使用验证集进行早停
    
    s_model.fit(tx, y_train)
        
    # 预测：分别计算T=1和T=0的情况，然后求差值得到CATE
    X_treat = np.concatenate([X_test, np.ones((X_test.shape[0], 1))], axis=1)  # [X, 1]
    X_control = np.concatenate([X_test, np.zeros((X_test.shape[0], 1))], axis=1)  # [X, 0]   
      
    y1_pred = s_model.predict(X_treat)   # E[Y|X, T=1]
    y0_pred = s_model.predict(X_control) # E[Y|X, T=0]
        
    cate_s_xgb = y1_pred - y0_pred  # CATE = E[Y|X, T=1] - E[Y|X, T=0]
    
    return cate_s_xgb

def T_learner(X_train, T_train, y_train,X_test,X_val, y_val, n_trials_xgb=50, xgb_opt_early_stopping_rounds=10, trial_patience=20):
    """
    T-learner实现 - 修改版避免数据泄露
    
    Args:
        X_full: 完整特征矩阵
        T_full: 完整处理变量
        y_full: 完整结果变量
        X_train: 训练集特征矩阵
        T_train: 训练集处理变量
        y_train: 训练集结果变量
        X_val: 验证集特征矩阵（仅用于最终评估）
        T_val: 验证集处理变量（仅用于最终评估）
        y_val: 验证集结果变量（仅用于最终评估）
        n_trials: Optuna试验次数
        xgb_opt_early_stopping_rounds: 超参数调优时的早停轮数
        xgb_train_early_stopping_rounds: 最终训练时的早停轮数
        trial_patience: trial级别早停耐心值
    """
    # 获取最优参数 - 使用训练集内部交叉验证
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val,y_val=y_val, 
                               n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    
    control_index = T_train == 0
    treatment_index = T_train == 1
       
    # 第一步：训练基础模型 μ0(X) 和 μ1(X)
    X_control = X_train[control_index]
    X_treatment = X_train[treatment_index]   
    y_control = y_train[control_index]
    y_treatment = y_train[treatment_index]

    mu0_model = XGBRegressor(**xgb_params, device='cuda')
    mu1_model = XGBRegressor(**xgb_params, device='cuda')
    
    # 训练control组模型 
    mu0_model.fit(X_control, y_control)
    # 训练treatment组模型    
    mu1_model.fit(X_treatment, y_treatment)

    cate_t_xgb = mu1_model.predict(X_test) - mu0_model.predict(X_test)

    return cate_t_xgb
    

def X_learner(X_train, T_train, y_train,X_test,X_val, T_val, y_val,n_trials_propensity=50, 
              n_trials_xgb=50, n_trials_lr=50, xgb_opt_early_stopping_rounds=10,y_opt_early_stopping_rounds=10, trial_patience=20):
    """
    X-learner实现
    
    Args:
        X: 特征矩阵
        T: 处理变量
        y: 结果变量
        X_val: 验证集特征矩阵
        T_val: 验证集处理变量
        y_val: 验证集结果变量
        n_trials_main: 主要模型的Optuna试验次数
        cv_folds_main: 主要模型的交叉验证折数
        n_trials_propensity: 倾向性模型的Optuna试验次数
        cv_folds_propensity: 倾向性模型的交叉验证折数
        n_trials_pre: prepare_y中模型的Optuna试验次数
        cv_folds_pre: prepare_y中模型的交叉验证折数
        run_times: 运行次数，用于计算平均值和标准差
        trial_patience: trial级别早停耐心值
    """
   
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                               n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    lr_params = model_prepare(model_name='lr', X_train=X_train, y_train=T_train, X_val=X_val, y_val=y_val,
                              n_trials=n_trials_lr, trial_patience=trial_patience)
   
    control_index = T_train == 0
    treatment_index = T_train == 1
        
        # 第一步：训练基础模型 μ0(X) 和 μ1(X)
    X_control = X_train[control_index]
    X_treatment = X_train[treatment_index]   
        
    mu0_pred, mu1_pred = prepare_y(X_train=X_train, T_train=T_train, y_train=y_train,X_val=X_val,T_val=T_val,y_val=y_val,
                                   n_trials=n_trials_propensity,y_opt_early_stopping_rounds=y_opt_early_stopping_rounds, trial_patience=trial_patience)
       
        # 训练倾向性得分模型 π(X) - 创建新实例
    propensity_model = LogisticRegression(**lr_params)
    propensity_model.fit(X_train, T_train)
    propensity_scores = propensity_model.predict_proba(X_test)[:, 1]
        
        # 第二步：计算pseudo outcomes
    D0 = mu1_pred[control_index] - y_train[control_index]
    D1 = y_train[treatment_index] - mu0_pred[treatment_index]   

        # 创建独立的模型实例用于τ1
    tau1_model = XGBRegressor(**xgb_params,device='cuda')
    tau1_model.fit(X_treatment, D1)

        # 创建独立的模型实例用于τ0
    tau0_model = XGBRegressor(**xgb_params,device='cuda')
    tau0_model.fit(X_control, D0) 
        # 最终预测：τ̂X(X) = (1-π(X))τ̂¹X(X) + π(X)τ̂⁰X(X)
    tau1_pred = tau1_model.predict(X_test)
    tau0_pred = tau0_model.predict(X_test)
        
        # 组合预测
    cate_x_xgb = (1 - propensity_scores) * tau1_pred + propensity_scores * tau0_pred
    
    return cate_x_xgb

def R_learner(X_train, T_train, y_train,X_test,X_val, y_val, 
              n_trials_xgb=50, n_trials_lr=50, xgb_opt_early_stopping_rounds=10, trial_patience=20):
    """
    R-learner实现
    
    Args:
        X: 特征矩阵
        T: 处理变量
        y: 结果变量
        X_val: 验证集特征矩阵
        T_val: 验证集处理变量
        y_val: 验证集结果变量
        n_trials_main: 主要模型的Optuna试验次数
        cv_folds_main: 主要模型的交叉验证折数
        n_trials_propensity: 倾向性模型的Optuna试验次数
        cv_folds_propensity: 倾向性模型的交叉验证折数
        trial_patience: trial级别早停耐心值
    """
   
    
    # 获取最优参数
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                               n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    lr_params = model_prepare(model_name='lr', X_train=X_train, y_train=T_train, X_val=X_val, y_val=y_val,
                              n_trials=n_trials_lr, trial_patience=trial_patience)

    # 训练倾向性得分模型 π(X) - 创建新实例
    propensity_model = LogisticRegression(**lr_params)
    propensity_model.fit(X_train, T_train)
    propensity_scores = propensity_model.predict_proba(X_train)[:, 1]
    
    # 训练条件均值模型 μ(X) 来近似 E[Y|X] - 创建新实例
    mu_model = XGBRegressor(**xgb_params,device='cuda')
    mu_model.fit(X_train, y_train)
    mu_pred = mu_model.predict(X_train) 
    
    # 计算残差
    # 结果残差 ξ = Y - μ(X)
    outcome_residual = y_train - mu_pred    
    # 处理残差 ν = T - π(X)  
    residual_t = T_train - propensity_scores     
    
    residual_y = outcome_residual / residual_t     
    # 创建独立的模型实例用于τ
    tau_model = XGBRegressor(**xgb_params,device='cuda')
    tau_model.fit(X_train, residual_y,sample_weight=residual_t**2)
    cate_r_xgb = tau_model.predict(X_test)
    
    return cate_r_xgb

def U_learner(X_train, T_train, y_train,X_test,X_val, y_val, 
              n_trials_xgb=50, n_trials_lr=50,xgb_opt_early_stopping_rounds=10, trial_patience=20):
    """
    U-learner实现
    
    Args:
        X: 特征矩阵
        T: 处理变量
        y: 结果变量
        X_val: 验证集特征矩阵
        T_val: 验证集处理变量
        y_val: 验证集结果变量
        n_trials_main: 主要模型的Optuna试验次数
        cv_folds_main: 主要模型的交叉验证折数
        n_trials_propensity: 倾向性模型的Optuna试验次数
        cv_folds_propensity: 倾向性模型的交叉验证折数
        trial_patience: trial级别早停耐心值
    """
    # 获取最优参数
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                               n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    lr_params = model_prepare(model_name='lr', X_train=X_train, y_train=T_train, X_val=X_val, y_val=y_val,
                              n_trials=n_trials_lr, trial_patience=trial_patience)
    
    # 训练倾向性得分模型 π(X) - 创建新实例
    propensity_model = LogisticRegression(**lr_params)
    propensity_model.fit(X_train, T_train)
    propensity_scores = propensity_model.predict_proba(X_train)[:, 1]
    
    # 训练条件均值模型 μ(X) 来近似 E[Y|X] - 创建新实例
    mu_model = XGBRegressor(**xgb_params,device='cuda')
    mu_model.fit(X_train, y_train)
    mu_pred = mu_model.predict(X_train)  
        
    # 计算残差
    # 结果残差 ξ = Y - μ(X)
    outcome_residual = y_train - mu_pred
    
    # 处理残差 ν = T - π(X)  
    treatment_residual = T_train - propensity_scores

    y_residual = outcome_residual / treatment_residual
    
    # 创建独立的模型实例用于τ
    tau_model = XGBRegressor(**xgb_params,device='cuda')    

    tau_model.fit(X_train, y_residual)
    cate_u_xgb = tau_model.predict(X_test)        
    
    return cate_u_xgb

def DR_learner(X_train, T_train, y_train,X_test,X_val, T_val, y_val, 
              n_trials_xgb=50, n_trials_lr=50, n_trials_propensity=50,
              xgb_opt_early_stopping_rounds=10,y_opt_early_stopping_rounds=10, trial_patience=20):
    """
    DR-learner (Doubly Robust)实现
    
    Args:
        X: 特征矩阵
        T: 处理变量
        y: 结果变量
        X_val: 验证集特征矩阵
        T_val: 验证集处理变量
        y_val: 验证集结果变量
        n_trials_main: 主要模型的Optuna试验次数
        cv_folds_main: 主要模型的交叉验证折数
        n_trials_propensity: 倾向性模型的Optuna试验次数
        cv_folds_propensity: 倾向性模型的交叉验证折数
        n_trials_pre: prepare_y中模型的Optuna试验次数
        cv_folds_pre: prepare_y中模型的交叉验证折数
        trial_patience: trial级别早停耐心值
    """
    # 获取最优参数
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                               n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    lr_params = model_prepare(model_name='lr', X_train=X_train, y_train=T_train, X_val=X_val, y_val=y_val,
                              n_trials=n_trials_lr, trial_patience=trial_patience)
    
    mu0_pred, mu1_pred = prepare_y(X_train=X_train, T_train=T_train, y_train=y_train,X_val=X_val,T_val=T_val,y_val=y_val,
                                   n_trials=n_trials_propensity,y_opt_early_stopping_rounds=y_opt_early_stopping_rounds, trial_patience=trial_patience)
    
    # 训练倾向性得分模型 π(X) - 创建新实例
    propensity_model = LogisticRegression(**lr_params)
    propensity_model.fit(X_train, T_train)
    propensity_scores = propensity_model.predict_proba(X_train)[:, 1]

    y_DR_0 = mu0_pred + ((1-T_train) / (1-propensity_scores)) * (y_train - mu0_pred)
    y_DR_1 = mu1_pred + (T_train / propensity_scores) * (y_train - mu1_pred)
    y_DR = y_DR_1 - y_DR_0

    # 创建独立的模型实例用于τ
    tau_model = XGBRegressor(**xgb_params,device='cuda')
    
    tau_model.fit(X_train, y_DR)
    cate_dr_xgb = tau_model.predict(X_test)

    return cate_dr_xgb

def RA_learner(X_train, T_train, y_train,X_test,X_val, T_val, y_val, n_trials_xgb=50, n_trials_propensity=50,
               xgb_opt_early_stopping_rounds=10,y_opt_early_stopping_rounds=10, trial_patience=20):
    """
    RA-learner (Regression Adjustment)实现
    
    Args:
        X: 特征矩阵
        T: 处理变量
        y: 结果变量
        X_val: 验证集特征矩阵
        T_val: 验证集处理变量
        y_val: 验证集结果变量
        n_trials_main: 主要模型的Optuna试验次数
        cv_folds_main: 主要模型的交叉验证折数
        n_trials_pre: prepare_y中模型的Optuna试验次数
        cv_folds_pre: prepare_y中模型的交叉验证折数
        trial_patience: trial级别早停耐心值
    """
    # 获取最优参数
    xgb_params = model_prepare(model_name='xgb', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                               n_trials=n_trials_xgb, xgb_opt_early_stopping_rounds=xgb_opt_early_stopping_rounds, trial_patience=trial_patience)
    
    mu0_pred, mu1_pred = prepare_y(X_train=X_train, T_train=T_train, y_train=y_train,X_val=X_val,T_val=T_val,y_val=y_val,
                                   n_trials=n_trials_propensity,y_opt_early_stopping_rounds=y_opt_early_stopping_rounds, trial_patience=trial_patience)    
    y_ra_0 = mu1_pred - y_train
    y_ra_1 = y_train - mu0_pred
    y_ra = T_train * y_ra_1 - (1-T_train) * y_ra_0

    # 创建独立的模型实例用于τ
    tau_model = XGBRegressor(**xgb_params,device='cuda')    

    tau_model.fit(X_train, y_ra)
    cate_ra_xgb = tau_model.predict(X_test)    
    
    return cate_ra_xgb




