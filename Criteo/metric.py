import numpy as np
from scipy import integrate
import warnings

def pehe(cate_pred, cate_true, k=None):
    """
    计算PEHE指标，按uplift得分排序后计算
    
    Args:
        cate_pred: 预测的CATE值
        cate_true: 真实的CATE值
        k: k值，如果为None则使用全部样本，否则使用前k%的样本（按排序后顺序）
    """
    cate_pred = np.array(cate_pred)
    cate_true = np.array(cate_true)
    
    # 按预测CATE降序排列
    desc_score_indices = np.argsort(cate_pred, kind="mergesort")[::-1]
    cate_pred = cate_pred[desc_score_indices]
    cate_true = cate_true[desc_score_indices]
    
    if k is not None:
        if k == 1.0:
            k_samples = len(cate_pred)
        else:
            k_samples = int(len(cate_pred) * k)
        cate_pred = cate_pred[:k_samples]
        cate_true = cate_true[:k_samples]
    
    return np.sqrt(np.mean(np.square(cate_pred - cate_true)))
   
def qini_coefficient_continuous_unscaled(y_true, uplift, treatment, k=None):
    
   y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
   desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
   y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]
   
   # 先确定k的值
   if k is None:
    k_samples = len(y_true)
   else:
    k_samples = int(len(y_true) * k)
   
   # 截断数据到前k个样本
   y_true = y_true[:k_samples]
   uplift = uplift[:k_samples]
   treatment = treatment[:k_samples]
   
   # 构建x：累计的靶向比例，从0到1
   # x[0] = 0 表示没有靶向任何个体（靶向深度为0）
   # x[i] = i/N 表示靶向前i个个体（按uplift降序排列）
   # x[N] = 1 表示靶向全部个体
   N = len(y_true)  # 现在N是截断后的长度   
   x = np.array(range(N+1)) / N
   treatment_cumulative = np.zeros(N)
   control_cumulative = np.zeros(N)
  # 根据对应的索引去做赋值
   treatment_cumulative[treatment == 1] = y_true[treatment == 1]
   control_cumulative[treatment == 0] = y_true[treatment == 0]
   
  #进行累加，得到累计结果率
   treatment_count_cumulative = np.zeros(N)
   control_count_cumulative = np.zeros(N)
   
   treatment_count_cumulative[treatment == 1] = 1
   N_T = np.cumsum(treatment_count_cumulative)
   control_count_cumulative[treatment == 0] = 1  
   N_C = np.cumsum(control_count_cumulative)

   F_T = np.cumsum(treatment_cumulative)
   F_C = np.cumsum(control_cumulative)
   
   with warnings.catch_warnings():
       warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered in divide')
       rate_for_F_C = np.where(N_C != 0, N_T / N_C, 0)
   
   qini_curve = F_T - (F_C * rate_for_F_C)
   qini_curve = np.concatenate([np.array([0]),qini_curve])

   g_1 = qini_curve[-1]
   Q_fun = qini_curve - (x * g_1)   
   qini_coefficient = integrate.trapz(Q_fun, x)
   
   return qini_coefficient / N 
   
  


def uplift_at_k_continuous(y_true, uplift, treatment, strategy, k=0.3):  

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    strategy_methods = ['overall', 'by_group']
    if strategy not in strategy_methods:
        raise ValueError(f'Uplift score supports only calculating methods in {strategy_methods},'
                         f' got {strategy}.')

    n_samples = len(y_true)
    order = np.argsort(uplift, kind='mergesort')[::-1]
    _, treatment_counts = np.unique(treatment, return_counts=True)
    n_samples_ctrl = treatment_counts[0]
    n_samples_trmnt = treatment_counts[1]

    k_type = np.asarray(k).dtype.kind
    if (k_type == 'i' and (k >= n_samples or k <= 0)
            or k_type == 'f' and (k <= 0 or k > 1)):  # 修改这里：k >= 1 改为 k > 1
        raise ValueError(f'k={k} should be either positive and smaller'
                         f' than the number of samples {n_samples} or a float in the '
                         f'(0, 1] range')  # 修改这里：添加了1

    if k_type not in ('i', 'f'):
        raise ValueError(f'Invalid value for k: {k_type}')

    if strategy == 'overall':
        if k_type == 'f':
            n_size = int(n_samples * k)
        else:
            n_size = k
        
        # ✅ 对连续值同样计算平均值
        ctrl_samples = y_true[order][:n_size][treatment[order][:n_size] == 0]
        trmnt_samples = y_true[order][:n_size][treatment[order][:n_size] == 1]
        
        # 添加检查避免空组
        if len(ctrl_samples) == 0:
            raise ValueError("No control samples in the first k observations")
        if len(trmnt_samples) == 0:
            raise ValueError("No treatment samples in the first k observations")
            
        score_ctrl = ctrl_samples.mean()
        score_trmnt = trmnt_samples.mean()
        
    else:  # strategy == 'by_group':
        if k_type == 'f':
            n_ctrl = int((treatment == 0).sum() * k)
            n_trmnt = int((treatment == 1).sum() * k)
        else:
            n_ctrl = k
            n_trmnt = k
            
        if n_ctrl > n_samples_ctrl:
            raise ValueError(f'With k={k}, the number of the first k observations'
                             ' bigger than the number of samples'
                             f'in the control group: {n_samples_ctrl}')
        if n_trmnt > n_samples_trmnt:
            raise ValueError(f'With k={k}, the number of the first k observations'
                             ' bigger than the number of samples'
                             f'in the treatment group: {n_samples_trmnt}')
        
        
        score_ctrl = y_true[order][treatment[order] == 0][:n_ctrl].mean()
        score_trmnt = y_true[order][treatment[order] == 1][:n_trmnt].mean()

    return score_trmnt - score_ctrl

def AUUC_continuous(y_true, uplift, treatment, k=None):
   y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
   desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
   y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]
   
   # 如果指定了k值，则只取前k%的样本
   if k is not None:
       if k == 1.0:
           k_samples = len(y_true)
       else:
           k_samples = int(len(y_true) * k)
       y_true = y_true[:k_samples]
       uplift = uplift[:k_samples]
       treatment = treatment[:k_samples]
   
   N = len(y_true)
   x = np.array(range(N+1)) / N
   treatment_cumulative = np.zeros(N)
   control_cumulative = np.zeros(N)
   
   # 根据对应的索引去做赋值
   treatment_cumulative[treatment == 1] = y_true[treatment == 1]
   control_cumulative[treatment == 0] = y_true[treatment == 0]
   # 进行累加，得到累计结果率
   F_T = np.cumsum(treatment_cumulative)
   F_C = np.cumsum(control_cumulative)
   
   treatment_count_cumulative = np.zeros(N)
   control_count_cumulative = np.zeros(N)
   # 计算N_T和N_C
   treatment_count_cumulative[treatment == 1] = 1
   N_T = np.cumsum(treatment_count_cumulative)
   control_count_cumulative[treatment == 0] = 1
   N_C = np.cumsum(control_count_cumulative)
   
   # 忽略除零警告，因为np.where会正确处理
   with warnings.catch_warnings():
       warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
       treatment_rate = np.where(N_T != 0, F_T / N_T, 0)
       control_rate = np.where(N_C != 0, F_C / N_C, 0)

   uplift_curve = (treatment_rate - control_rate) *  (N_T + N_C)
   uplift_curve = np.concatenate([np.array([0]),uplift_curve])
   
   # AUUC是uplift曲线下的面积（简单求和，因为x轴是等间距的）
   auuc = integrate.trapz(uplift_curve, x)
   return auuc / N
def ate_error(cate_pred, cate_true, k=None):
    """
    计算ATE error指标，按uplift得分排序后计算
    
    Args:
        cate_pred: 预测的CATE值
        cate_true: 真实的CATE值
        k: k值，如果为None则使用全部样本，否则使用前k%的样本（按排序后顺序）
    """
    cate_pred = np.array(cate_pred)
    cate_true = np.array(cate_true)
    
    # 按预测CATE降序排列
    desc_score_indices = np.argsort(cate_pred, kind="mergesort")[::-1]
    cate_true = cate_true[desc_score_indices]
    
    if k is not None:
        if k == 1.0:
            k_samples = len(cate_pred)
        else:
            k_samples = int(len(cate_pred) * k)        
        cate_true = cate_true[:k_samples]
    
    return np.mean(cate_true)

if __name__ == "__main__":
    small_y_true = np.array([1.5, 0.6, 1.7, 0.3, 1.5, 0.2, 1.6, 0.9])
    small_treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # 一半处理组，一半对照组
    small_uplift = np.array([0.8, 0.7, 0.3, 0.6, 0.5, 0.3, 0.5, 0.4])  # 排序后的提升分数
    auuc = AUUC_continuous(small_y_true, small_uplift, small_treatment, k=1.0)
    print(auuc)
    
    








