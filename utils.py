import torch
import torch.nn as nn
from pesq import pesq
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1.2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    
def pesq_loss(clean, noisy, sr=16000):
    try:
        return pesq(sr, clean, noisy, 'wb')
    except Exception as e:
        # error can happen due to silent period
        return -1

ctx = mp.get_context('spawn')
pesq_executor = ProcessPoolExecutor(max_workers=2, mp_context=ctx)

def batch_pesq(clean_list, noisy_list):
    # 保证返回的结果顺序与输入完全一致
    results = pesq_executor.map(pesq_loss, clean_list, noisy_list)
    
    pesq_score = np.array(list(results))
    
    if -1 in pesq_score:
        return None
        
    pesq_score = (pesq_score + 0.5) / 5.0
    
    return torch.FloatTensor(pesq_score).cuda(non_blocking=True)