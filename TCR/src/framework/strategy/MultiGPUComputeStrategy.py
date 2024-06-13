"""
Multi GPU Compute Strategy using MPI for communication
"""

import os
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
from src.framework.strategy.Strategy import Strategy
from src.framework.GASProgram import GASProgram
from src.type.Subgraph import Subgraph
from src.framework.partition.Partition import Partition
import numpy as np
import torch

class MultiGPUComputeStrategy(Strategy):
    def __init__(self, prog: GASProgram, partition: Partition):
        # all_vertex_data: np array
        # self.all_vertex_data = prog.vertex_data.numpy()
        # add edge_data
        self.prog = prog
        # 全局的激活节点掩码 np
        self.activated_vertices_mask = prog.activated_vertex_mask.numpy()
        # self.all_vertex_data_np = prog.vertex_data.numpy()
        # 本地的激活节点掩码 np
        # self.changed_vertices_mask = np.zeros_like(self.activated_vertices_mask, dtype=np.bool)
        # self.changed_vertices_mask = self.changed_vertices_mask.unsqueeze(1)
        # changed_vertices_mask 是否能做压缩处理？ 或者在需要时才调入GPU
        # 压不压缩无所谓
        # self.changed_vertices_mask[subgraph.sub_vertices] = True
        # 记录当前迭代次数
        self.curr_iter = 0
        self.partition = partition


    def compute(self):
        """
        执行本节点计算
        1. 获取本地激活节点
        2. 激活节点计算
        3. 激活节点数据全局更新、激活节点掩码全局更新
        """
        # rank 0 进程切割并分发subgraph
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        prog = self.prog
        device = f'cuda:{rank}'
        prog.to(device)
        
        if rank == 0:
            graph = prog.graph
            self.partition.set_graph(graph)
            partitions = self.partition.generate_partitions()
            subgraph = []
            for (p, i), v in partitions:
                orig_to_sub_vertices = torch.zeros_like(graph.vertices)
                orig_to_sub_vertices[v] = torch.arange(v.numel())
                new_subgraph = Subgraph(p, v, i, orig_to_sub_vertices)
                # logging.info('v {}'.format(v))
                subgraph.append(new_subgraph)
            all_vertex_data_np = prog.vertex_data.cpu().numpy()
        else:
            subgraph = None
            all_vertex_data_np = None
        # scatter subgraph
        subgraph = comm.scatter(subgraph, root=0)
        all_vertex_data_np = comm.bcast(all_vertex_data_np, root=0)

        prog.set_graph(subgraph)
        subgraph.to(device)
        # self.changed_vertices_mask = torch.zeros_like(self.activated_vertices_mask)
        self.changed_vertices_mask = np.zeros_like(all_vertex_data_np, dtype=np.bool)
        mask = np.ones((prog.vertex_data.shape[1],), dtype=np.bool)  \
            if len(prog.vertex_data.shape) == 2 else np.ones(1, dtype=np.bool)
        self.changed_vertices_mask[subgraph.sub_vertices.cpu().numpy()] = mask
        
        while not np.all(self.activated_vertices_mask == 0):
            # 循环入口执行同步操作
            # local_vertex_data = np.where(self.changed_vertices_mask, all_vertex_data_np, np.zeros_like(all_vertex_data_np))
            logging.info('process {} iter {} activated vertices {}'.format(rank, prog.curr_iter, self.activated_vertices_mask))
            local_vertex_data = np.where(self.changed_vertices_mask, all_vertex_data_np, np.zeros_like(all_vertex_data_np))
            comm.Allreduce(local_vertex_data, all_vertex_data_np, op=MPI.SUM)
            # 生成vertex_data tensor 并放入对应设备 进行后续计算
            prog.vertex_data = torch.from_numpy(all_vertex_data_np).to(prog.device)

            # 获取激活节点
            local_activated_vertices_mask = self.activated_vertices_mask[subgraph.sub_vertices.cpu().numpy()]
            # 激活节点索引 也是 subgraph vertex id
            local_activated_vertices_idx = np.where(local_activated_vertices_mask)[0]
            if local_activated_vertices_idx.shape[0] == 0: # 该GPU当前无激活节点
                prog.curr_iter += 1
                continue

            # 激活节点列表 origin id tensor on GPU array 
            local_activated_vertices = subgraph.sub_vertices[local_activated_vertices_idx].to(prog.device)
            logging.info('process {} local_activated_vertices {}'.format(rank, local_activated_vertices))
            # np array to tensor
            # local_activated_vertices = torch.from_numpy(local_activated_vertices).to(prog.device)

            # 本地计算
            logging.info('process {} local compute {}'.format(rank, self.curr_iter))
            
            # subgraph 的 in_nbrs_csr(vertices) 方法中的vertices是origin id
            # logging.info('vertices device {}'.format(local_activated_vertices.device))
            self.g_nbrs, self.g_edges, self.g_ptr = prog.gather_nbrs(local_activated_vertices)
            # logging.info('devices {}  {}'.format(self.g_nbrs.device, self.g_ptr.device))
            # gather data
            gd, g_ptr = prog.gather(local_activated_vertices, self.g_nbrs, self.g_edges, self.g_ptr)
            # sum data
            logging.info('process {} gd {}'.format(rank, gd))
            logging.info('process {} g_ptr {}'.format(rank, g_ptr))
            gsum = prog.sum(gd, g_ptr)
            logging.info('process {} gsum {}'.format(rank, gsum))
            # apply data
            apply_d, apply_mask = prog.apply(local_activated_vertices, gsum)
            logging.info('process {} apply_d {} apply_mask {}'.format(rank, apply_d, apply_mask))
            if apply_d is not None and apply_mask is not None:
                prog.vertex_data[local_activated_vertices] = torch.where(apply_mask, apply_d, prog.vertex_data[local_activated_vertices])
            logging.info('process {} prog.vertex_data {}'.format(rank, prog.vertex_data))
            # 更新 节点数据 numpy array
            all_vertex_data_np = prog.vertex_data.cpu().numpy()
            # scatter data and activated vertices
            self.s_nbrs, self.s_edges, self.s_ptr = prog.scatter_nbrs(local_activated_vertices)
            s_data, s_mask = prog.scatter(local_activated_vertices, self.s_nbrs, self.s_edges, self.s_ptr, apply_d)
            if not prog.not_change_activated:
                # 全局激活节点更新 可以用 bool 类型的numpy数组直接做 AllReduce LOR操作
                next_activated_vertices_mask = prog.next_activated_vertex_mask.cpu().numpy()
                comm.Allreduce(next_activated_vertices_mask, self.activated_vertices_mask, op=MPI.LOR)
            else:
                prog.not_change_activated = False
                prog.curr_iter += 1
                continue
            prog.curr_iter += 1
        
        # 获取最终结果
        all_vertex_data_np = prog.vertex_data.cpu().numpy()
        local_vertex_data = np.where(self.changed_vertices_mask, all_vertex_data_np, np.zeros_like(all_vertex_data_np))
        comm.Allreduce(local_vertex_data, all_vertex_data_np, op=MPI.SUM)
        # 生成vertex_data tensor 并放入对应设备
        prog.vertex_data = torch.from_numpy(all_vertex_data_np).to(prog.device)

        return prog.vertex_data, None




