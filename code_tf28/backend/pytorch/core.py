import torch
def loss_distance(index_sub, label_index_sub,device, decimal_places=8,):
    scaling_factor = 10 ** decimal_places
    XA = torch.tensor(index_sub, device=device, dtype=torch.int64)
    XB = torch.tensor(label_index_sub, device=device, dtype=torch.int64)
    # 在維度 1 上增加一個維度
    XA = XA.unsqueeze(1)
    # 在維度 0 上增加一個維度
    XB = XB.unsqueeze(0)
    # 廣播機制：將 XA 和 XB 的大小擴展為相同的形狀
    XA = XA.expand(XA.size(0), XB.size(1), XA.size(2))
    XB = XB.expand(XA.size(0), XB.size(1), XB.size(2))
    a_int = (XA * scaling_factor).long()
    b_int = (XB * scaling_factor).long()
    distances = torch.nn.functional.pairwise_distance(a_int, b_int)
    distances_float = distances.float() / scaling_factor
    # 將結果移回 CPU
    loss_min = (torch.round(distances_float.amin(1), decimals=5).cpu())
    return loss_min