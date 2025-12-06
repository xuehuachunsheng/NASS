import logging
import random
import numpy as np
import torch
import torch.nn as nn
import faiss

def set_seed(seed=0):  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def sup_loss(logits, labels):
    # logits are outputed by the linear classifier
    # labels are label indices, without one-hot encoding
    loss = nn.functional.cross_entropy(logits, labels, reduction="mean")
    return loss

def true_sample_idx_each_class(trainset, _labeled_idx, CONF):
    gt_class_idx = [[] for _ in range(CONF.n_classes)]
    for i in _labeled_idx:
        x, y = trainset[i]
        gt_class_idx[y.item()].append(i)
    return gt_class_idx

def pred_probs(model, train_ds, unlabeled_idx, CONF):
    # here the unlabeled pool is actually the original train set.
    model.eval()
    c_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONF.bs, shuffle=False, num_workers=0)
    predicted_probs = [] # NxC
    with torch.no_grad():
        for x_batch, _ in c_loader:
            x_batch = x_batch.to(CONF.DEVICE)
            logits = model(x_batch)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            predicted_probs.extend(probs.tolist())

    predicted_probs_mapping = {}
    for id in unlabeled_idx:
        predicted_probs_mapping[id] = predicted_probs[id]
    return predicted_probs_mapping

def stratify(unlabeled_probs, CONF):
    C = CONF.n_classes
    stratified_idx = [[] for _ in range(C)]
    for id in unlabeled_probs:
        prob = unlabeled_probs[id]
        label = np.argmax(prob)
        stratified_idx[label].append(id)
    return stratified_idx

def compute_bc(n_cs, CONF):
    if CONF.known_population:
        return compute_bc_knownpopulation(n_cs, CONF.population_ratio,CONF)
    return compute_bc_unkownpoplation(n_cs,CONF)

def compute_bc_unkownpoplation(n_cs, CONF):
    nc = np.asarray(n_cs)
    C = CONF.n_classes
    q_budget = CONF.B

    mean_nc = np.mean(nc)
    C_hat = np.where(nc - mean_nc < q_budget / C)[0] # C_hat
    assert len(C_hat) != 0
    bc = np.zeros(C, dtype=np.int32)
    for i in range(C): 
        if nc[i] - mean_nc < q_budget / C:
            bc[i] = np.round(mean_nc - nc[i] + q_budget / len(C_hat))
    
    # 严格控制查询预算
    t_Bi = int(np.sum(bc))
    while t_Bi < q_budget: # 如果实际小于预期budget，则随机选择C_hat中的类别提高其查询数量，直到满足当前budget
        tc = np.random.randint(low=0, high=len(C_hat))
        bc[C_hat[tc]] += 1
        t_Bi += 1
    while t_Bi > q_budget: # # 如果实际大于预期budget，则随机选择C_hat中的类别降低其查询数量，直到满足当前budget
        tc = np.random.randint(low=0, high=len(C_hat))
        if bc[C_hat[tc]] > 0: # 不允许bc小于0
            bc[C_hat[tc]] -= 1
            t_Bi -= 1
    assert t_Bi == q_budget
    return bc

def compute_bc_knownpopulation(n_cs, population_ratio, CONF):
    nc = np.asarray(n_cs)
    n = np.sum(nc)
    C = CONF.n_classes
    q_budget = CONF.B
    P = population_ratio
    #mean_nc = np.mean(nc)
    #C_hat = np.where(nc - mean_nc < q_budget / C)[0] # C_hat
    C_hat = []
    for i in range(C):
        if nc[i] < (q_budget + n) * P[i]:
            C_hat.append(i)
    PC_hat = np.sum([population_ratio[c] for c in C_hat])
    assert len(C_hat) != 0
    bc = np.zeros(C, dtype=np.int32)
    for i in range(C): 
        if nc[i] < (q_budget + n) * P[i]:
            bc[i] = np.round(n*P[i] - nc[i] + q_budget*P[i] / PC_hat)
    
    # 严格控制查询预算
    t_Bi = int(np.sum(bc))
    while t_Bi < q_budget: # 如果实际小于预期budget，则随机选择C_hat中的类别提高其查询数量，直到满足当前budget
        tc = np.random.randint(low=0, high=len(C_hat))
        bc[C_hat[tc]] += 1
        t_Bi += 1
    while t_Bi > q_budget: # # 如果实际大于预期budget，则随机选择C_hat中的类别降低其查询数量，直到满足当前budget
        tc = np.random.randint(low=0, high=len(C_hat))
        if bc[C_hat[tc]] > 0: # 不允许bc小于0
            bc[C_hat[tc]] -= 1
            t_Bi -= 1
    assert t_Bi == q_budget
    return bc

def compute_uncertainty(probs_mapping, method):
    info_mapping = {}
    for sample_id in probs_mapping:
        prob = np.asarray(probs_mapping[sample_id])
        info = None
        if method == 'entropy': # 越大越好
            info = -np.sum((prob + 1e-6) * np.log(prob + 1e-6))
        elif method == 'leastconfidence': # 越小越好
            info = np.max(prob) 
        elif method == "margin": # 越小越好
            prob = np.sort(prob)
            info = prob[-1] - prob[-2]
        info_mapping[sample_id] = info
    return info_mapping

def weighted_prec_rec_f1_detailed(y_true, y_pred, class_weights, CONF, CONFSP):
    """
    返回详细结果的加权指标计算
    
    Returns:
    dict: 包含加权指标和每个类别详细指标的结果
    exclude_class_idx: 需要排除的类别id
    """
    classes = list(range(CONF.n_classes))
    
    class_weights = np.array(class_weights, copy=True)
    # 归一化权重
    normalized_weights = np.asarray(class_weights) / np.sum(class_weights)
    precisions = []
    recalls = []
    f1_scores = []
    
    # 存储每个类别的详细结果
    class_results = {}
    
    for i, cla in enumerate(classes):
        y_true_binary = (np.array(y_true) == cla).astype(int)
        y_pred_binary = (np.array(y_pred) == cla).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        precision_cls = tp / (tp + fp + 1e-9)
        recall_cls = tp / (tp + fn + 1e-9)
        f1_cls = 2 * (precision_cls * recall_cls) / (precision_cls + recall_cls + 1e-9)
        
        precisions.append(precision_cls)
        recalls.append(recall_cls)
        f1_scores.append(f1_cls)
        
        class_results[cla] = {
            'precision': precision_cls,
            'recall': recall_cls,
            'f1': f1_cls,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'weight': normalized_weights[i]
        }
    
    # 计算加权指标
    weighted_precision = sum(p * w for p, w in zip(precisions, normalized_weights))
    weighted_recall = sum(r * w for r, w in zip(recalls, normalized_weights))
    weighted_f1 = sum(f * w for f, w in zip(f1_scores, normalized_weights))
    
    class_weights[CONFSP.exclude_class_idx] = 0
    normalized_weights = np.asarray(class_weights) / np.sum(class_weights)
    weighted_precision_nw = sum(p * w for p, w in zip(precisions, normalized_weights))
    weighted_recall_nw = sum(r * w for r, w in zip(recalls, normalized_weights))
    weighted_f1_nw = sum(f * w for f, w in zip(f1_scores, normalized_weights))
    
    return {
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'weighted_precision_nw': weighted_precision_nw,
        'weighted_recall_nw': weighted_recall_nw,
        'weighted_f1_nw': weighted_f1_nw,
        'class_details': class_results,
        'per_class_precision': precisions,
        'per_class_recall': recalls,
        'per_class_f1': f1_scores,
    }


def cleanup_logger(logger_name):
    """
    完全清理指定名称的logger
    
    Parameters:
    logger_name: logger的名称
    """
    # 获取logger
    logger = logging.getLogger(logger_name)
    
    # 1. 移除所有handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        try:
            handler.close()
        except:
            pass
    
    # 2. 移除所有filter
    logger.filters.clear()
    
    # 3. 从manager中移除logger
    if logger_name in logging.Logger.manager.loggerDict:
        del logging.Logger.manager.loggerDict[logger_name]
    
    print(f"Logger '{logger_name}' 已清理")

# 获取一个ndarray的排名array
def get_column_ranks(data, ascending=False):
    """
    获取每列的排名
    ascending: True为升序排名(值小排名高), False为降序排名(值大排名高)
    """
    # 获取排序后的索引
    if ascending:
        sorted_indices = np.argsort(data, axis=0)  # 升序排序索引
    else:
        sorted_indices = np.argsort(-data, axis=0)  # 降序排序索引
    
    # 创建排名数组
    ranks = np.zeros_like(data)
    n_rows = data.shape[0]
    
    # 为每个位置分配排名
    for col in range(data.shape[1]):
        ranks[sorted_indices[:, col], col] = np.arange(1, n_rows + 1)
    
    return ranks.astype(int)

def find_all_neighbors_in_X(X, epsilon, exclude_self=True):
    """
    使用 FAISS range_search 找出 X 中每个样本在 L2 距离 ε 内的所有邻居下标。
    
    Args:
        X: numpy.ndarray, shape (n, d), dtype=float32
        epsilon: float, L2 距离阈值（真实距离，非平方）
        exclude_self: bool, 是否排除自身（即 j != i）
    
    Returns:
        neighbor_indices: list of np.ndarray, 长度为 n，
                          neighbor_indices[i] 是 X[i] 的邻居下标数组
    """
    assert X.dtype == np.float32, "X must be float32"
    n, d = X.shape
    # 1. 构建 FAISS 索引（使用 L2）
    index = faiss.IndexFlatL2(d)
    index.add(X)  # 将整个 X 作为数据库
    # 2. 设置平方半径（FAISS L2 使用平方距离！）
    radius_sq = epsilon ** 2
    # 3. 执行 range_search：用 X 自身作为查询
    lims, distances, indices = index.range_search(X, radius_sq)
    # 4. 解析结果：按查询拆分
    neighbor_indices = []
    for i in range(n):
        start = lims[i]
        end = lims[i + 1]
        neighbors = indices[start:end]
        if exclude_self:
            # 排除自身（注意：可能有重复或浮点误差，但通常 self 距离=0）
            neighbors = neighbors[neighbors != i]
        neighbor_indices.append(neighbors)
    
    return neighbor_indices