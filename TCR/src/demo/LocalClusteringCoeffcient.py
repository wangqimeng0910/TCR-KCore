"""
Label Propagation for semi-supervised learning. Assuming undirected graphs.
"""
import sys
# sys.path.append('/root/Project/TCRGraph')
sys.path.append('/home/lames/TCRGraph/')
from src.type.Graph import Graph
import logging
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.GASProgram import GASProgram
from src.type.CSRGraph import CSRGraph
import torch
import argparse
import time
import random
from src.framework.helper import batched_csr_selection
from torch_scatter import segment_csr

class LocalClusteringCoeffcient(GASProgram):
    def __init__(self, graph: Graph, vertex_data_type=torch.float32, edge_data_type=torch.float32, vertex_data=None, edge_data=None, start_from=None, nbr_update_freq=0):
        # vertex_data: [num_vertices]. Each vertex has a value Ci. By default each vertex has C == 0.
        if vertex_data is None:
            vertex_data = torch.ones((graph.num_vertices,), dtype=vertex_data_type)
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data, start_from, nbr_update_freq)
    
    def __init__(self, graph: Graph, vertex_data_type=torch.float32, edge_data_type=torch.float32, vertex_data=None, edge_data=None, start_from=None, nbr_update_freq=0):
        # vertex_data: [num_vertices]. Each vertex has a value Ci. By default each vertex has C == 0.
        if vertex_data is None:
            vertex_data = torch.ones((graph.num_vertices,), dtype=vertex_data_type)
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data, start_from, nbr_update_freq)
    
    def gather(self, vertices, nbrs, edges, ptr):
        """

        Returns:
            gd: gathered data
            gd_ptr: ptr of gd
        """
        # 获取出度信息
        degree = self.graph.out_degree(vertices)
        nbrs_clone = nbrs.clone()

        # 将节点序号替换为其出度
        nbrs_clone = degree[nbrs_clone]
        # 获取每个节点的 `size`
        size = segment_csr(nbrs_clone, ptr, reduce='sum')
        # 计算 `new_ptr`
        gd_ptr = torch.cat((torch.zeros((1,), dtype=torch.long, device=self.device), torch.cumsum(size, dim=0)))
        # logging.info('gd_ptr: {}'.format(gd_ptr))
        # 计算 `gd`
        # gd = torch.zeros((gd_ptr[-1],), dtype=self.vertex_data_type, device=self.device)
        tensors = [nbrs[ptr[value] : ptr[value + 1]] for value in nbrs]
        gd = torch.cat(tensors, dim=0)
        return gd, gd_ptr
    
    def sum(self, gathered_data, ptr):
        """
        Args:
            gathered_data: 
            ptr: gd_ptr
        Returns:
            gsum: λ(G)
        """
        # function intersection is used to calculate the num of intersection elements of two tensors
        intersection = lambda tensor1, tensor2: torch.tensor([torch.sum((tensor1.view(-1, 1) == tensor2).any(dim=0)).item() / 2], dtype=torch.int32, device=self.device)
        g_nbrs, _, g_ptr = self.gather_nbrs(self.graph.vertices)
        tensors = [intersection(g_nbrs[g_ptr[i] : g_ptr[i + 1]], 
                                gathered_data[ptr[i] : ptr[i + 1]])
                   for i in range(len(ptr) - 1)]   
    
        # logging.info('tensors: {}'.format(tensors))
        # logging.info('gathered_data_device: {}'.format(gathered_data.device))
        gsum = torch.cat(tensors, dim=0)
        # logging.info('gsum_device: {}'.format(gsum.device))
        return gsum

    def apply(self, vertices, gathered_sum):
        """
        Compute `Ci` for vertices

        Returns:
            apply_d:
            apply_mask:
        """
        # logging.info('gathered_sum: {}'.format(gathered_sum.device))
        # logging.info('out_degree: {}'.format(self.graph.out_degree(vertices).device))
        apply_d = gathered_sum * 2 / self.graph.out_degree(vertices) \
                                                      / (self.graph.out_degree(vertices) - 1)
        apply_mask = torch.ones(len(vertices), dtype=torch.bool, device=self.device)
        # 将 `nan` 替换为 0.0  `nan` 代表出度为 1 进行上述运算时出现的除零错误
        apply_d[torch.isnan(apply_d)] = 0.0
        return apply_d, apply_mask
    
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        """
        No need to scatter
        """
        return None, None
    
    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr
    
    def scatter_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph, _ = CSRGraph.read_graph(args.graph, split=None)
    graph.pin_memory()
    
    lcc = LocalClusteringCoeffcient(graph)
    strategy = SimpleStrategy(lcc)
    if args.cuda:
        lcc.to('cuda')
    
    t1 = time.time()
    data,_ = lcc.compute(strategy)
    t2 = time.time()
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results

    # path = '/root/Project/TCRGraph/profile/result/log'
    # with open(path, 'a') as f:
    #     data = "LCC  data: {} time: {} \n".format(args.graph, t2 - t1)
    #     f.write(data)

if __name__ == '__main__':
    main()
