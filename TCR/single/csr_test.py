import torch
from torch_scatter import segment_csr, segment_coo, scatter_add
import time
import random
if __name__ == "__main__":
    size = 10000000    
    

    # src = torch.rand(size, dtype=torch.float16)
    # index = 0
    # length = 100
    # indexs = torch.arange(10000, dtype=torch.int64) 
    # indexs = indexs.repeat_interleave(length)
    
    # t1 = time.time()
    # for i in range(10):
    #     ans = segment_coo(src, indexs, dim_size=10001)
    # # torch.cuda.synchronize()
    # t2 = time.time()
    # print(t2 - t1)

    # indexs = indexs[torch.randperm(indexs.size(0))]
    # t1 = time.time()
    # for i in range(10):
    #     ans = segment_coo(src, indexs, dim_size=10001)
    # # torch.cuda.synchronize()
    # t2 = time.time()
    # print(t2 - t1)

    tensor = torch.rand(size, dtype=torch.float16).cuda()
    t1 = time.time()
    ans = tensor[3000000:]
    torch.cuda.synchronize()
    t2 = time.time()
    print(t2 - t1)