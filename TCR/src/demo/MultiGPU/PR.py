import sys
import torch
import argparse
import time
import logging
from torch_scatter import segment_csr
from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack
import cupy as cp
from cupy.cuda import nccl
from cupy import cuda
import os
import torch.multiprocessing as mp
sys.path.append('/root/Project/TCRGraph/')
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.framework.helper import batched_csr_selection
from src.framework.strategy.MultiGPUComputeStrategy import MultiGPUComputeStrategy
from src.framework.partition.GeminiPartition import GeminiPartition
from src.type.Subgraph import Subgraph
import torch.distributed as dist
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class PRComputeOnGPU(mp.Process):
    def __init__(self, graph: Subgraph, rank, size, all_vertex_num, result_queue=None, **kwarges) -> None:
        super().__init__(**kwarges)
        self.rank = rank
        self.size = size
        self.device = f'cuda:{rank - 1}'
        self.result_queue = result_queue
        # subgraph 加一个mask 指示哪些点是本地节点，哪些不是 这个 mask是写死
        self.graph = graph
        self.graph.to(device=self.device)
        self.curr_iter = 0
        
        self.local_vertex_data = torch.ones(graph.num_vertices, dtype=torch.float32).to(self.device)
        self.non_local_vertex_data = torch.zeros(self.graph.non_local_vertex_num, dtype=torch.float32).to(self.device)  

        # 所有的节点数据 包括本地和非本地
        self.all_vertex_data = torch.ones(graph.num_vertices + self.graph.non_local_vertex_num, dtype=torch.float32).to(self.device)
        # 用于 segment_csr的数据
        self.data = torch.ones_like(self.graph.csc.csr.columns, dtype=torch.float32).to(self.device)
        self.frac = 1 / self.graph.csr.out_degrees * 0.85
        self.dead_nodes = torch.where(self.graph.csr.out_degrees == 0)[0]
        self.local_vertex_data[self.dead_nodes] = 0.85 / all_vertex_num
        # 出度为0的节点的frac
        self.xs = 0.85 / all_vertex_num
        self.row_ptr = self.graph.csc.csr.row_ptr.to(self.device)
        # columns 数据重编码
        self.data_index = self.graph.data_index.to(self.device)
    def init_process(self, backend='nccl'):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend, rank=self.rank, world_size=self.size)

    def run(self):
        # init 
        self.init_process()
        group = dist.new_group([i for i in range(self.size)])
        # 消除初始化环境的影响
        tensor = torch.tensor([self.rank]).to(self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)

        # do computation
        while self.curr_iter < 15:
            logging.info('rank {} iter {}'.format(self.rank, self.curr_iter))

            # 获取 nonlocal vertex data
            dist.recv(self.non_local_vertex_data, src=0)
            # 计算
            self.all_vertex_data[-self.graph.non_local_vertex_num:] = self.non_local_vertex_data
            # logging.info('rank {} recv nonlocal vertex data'.format(self.rank))
            x = torch.sum(self.all_vertex_data[self.dead_nodes]) * self.frac + 0.15
            self.data = self.all_vertex_data[self.data_index]
            self.all_vertex_data[:self.graph.num_vertices] = segment_csr(self.data, self.row_ptr, reduce='sum') + x

            # send local vertex data
            dist.send(self.all_vertex_data[:self.graph.num_vertices], dst=0)
        
class Master(mp.Process):
    def __init__(self, rank, size, receive_masks, send_masks, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rank = rank
        self.size = size
        self.receive_masks = receive_masks
        self.send_masks = send_masks
        

if __name__ == '__main__':
    graph, _, _ = CSRCGraph.read_graph('/root/Project/TCRGraph/data/graph.txt')
    graph.pin_memory()
    partition = GeminiPartition(graph, 2, 0.15)
    partitions = partition.generate_partitions()
    for (p, i), v in partitions:
        print(type(p))
        # print(i)
        # print(v)
        print(p.csr.row_ptr)
        print(p.csr.columns)
        print(p.csr.out_degrees)