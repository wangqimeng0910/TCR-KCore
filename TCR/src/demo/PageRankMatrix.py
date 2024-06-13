"""
Implementation of PageRank algorithm using the GAS framework.
"""

import sys
import torch
import argparse
import time
import logging
from viztracer import VizTracer
from torch_scatter import segment_csr
sys.path.append('/root/Project/TCRGraph/')
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.framework.helper import batched_csr_selection
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.partition.GeminiPartition import GeminiPartition
from src.framework.strategy.MultiGPUStrategyByNCCL import MultiGPUStrategyByNCCL
from src.framework.strategy.MultiGPUStrategyByCupyAndNCCL import MultiGPUStrategyByCupyAndNCCL
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class PageRank(GASProgram):
    def __init__(self, graph: CSRCGraph, vertex_data_type=torch.float32, edge_data_type=torch.float32, num_iter=25,
                 **kwargs):
        # vertex_data: [num_vertices, 2]. rank and delta
        # vertex_data = torch.ones((graph.num_vertices, 2), dtype=vertex_data_type)
        vertex_data = torch.ones(graph.num_vertices, dtype=vertex_data_type)
        self.vertex_delta = torch.ones(graph.num_vertices, dtype=vertex_data_type)
        # vertex_data /= graph.num_vertices
        self.UPDATE_THRESHOLD = 0.001
        self.ITER_THRESHOLD = num_iter
        self.out_degree = None
        self.matrix = None  # 转移矩阵
        super().__init__(graph, vertex_data=vertex_data, vertex_data_type=vertex_data_type,
                         edge_data_type=edge_data_type, **kwargs)

    def gather(self, vertices, nbrs, edges, ptr):

        if self.matrix == None:
            col_indices = self.graph.csc.csr.row_ptr
            rows = self.graph.csc.csr.columns
            values = self.graph.out_degree(rows)
            values = 1 / values
            size = [self.graph.num_vertices, self.graph.num_vertices]
            self.matrix = torch.sparse_csr_tensor(col_indices, rows, values, size)
            # 没有出度的节点, 如果不考虑无出度的节点，会导致结果较大误差
            # 默认没有出度的节点到所有节点的转移概率为 1 / num_vertices
            self.deads = torch.where(self.graph.csr.out_degrees == 0)[0]
            
        # 所有无出度节点的 pr值之和 
        data = torch.sum(self.vertex_data[self.deads]) / self.graph.num_vertices
        return data, ptr

    def sum(self, gathered_data, ptr):
        """
        gathered_data is the sum of dead nbrs' pr value
        """
        sum_data = self.matrix.matmul(self.vertex_data) + gathered_data
        return sum_data


    def apply(self, vertices, gathered_sum):
        """
        Returns:
            apply_data: 
            apply_mask: 
        """
        
        rnew = 0.15 + 0.85 * gathered_sum
        delta = torch.abs(rnew - self.vertex_data)
        self.vertex_data = rnew
        self.vertex_delta = delta
        return None, None

    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        """
        决定是否停止循环, 如果delta小于阈值则停止循环
        但不是每一轮迭代都判断(太耗时), 每隔10轮次判断一下
        """
        if self.curr_iter < self.ITER_THRESHOLD - 1:
            if self.curr_iter != 0 and self.curr_iter % 1 == 0:
                # selected = torch.where(self.vertex_delta > self.UPDATE_THRESHOLD)[0]
                # logging.info('selected: {}'.format(selected.shape[0]))
                pass
        else:
            self.is_quit = True
          
        return None, None

    def gather_nbrs(self, vertices):
        
        in_nbrs, ptr = self.graph.all_in_nbrs_csr()
        return in_nbrs, None, ptr

    def scatter_nbrs(self, vertices):
        
        out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        return out_nbrs, None, ptr

    def quit(self):
        # t1 = time.time()
        # max_val = torch.max(self.vertex_delta)
        # t2 = time.time()
        # logging.info('quit time {}'.format(t2 - t1))
        # if max_val < self.UPDATE_THRESHOLD:
        #     return True
        # else:
        #     return False

        selected = torch.where(self.vertex_delta > self.UPDATE_THRESHOLD)[0]
        torch.cuda.synchronize()
        if selected.shape[0] > 0:
            return False
        else:
            return True
        # ans = torch.all(self.vertex_delta < self.UPDATE_THRESHOLD)
        # torch.cuda.synchronize()
        # return ans
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()

    print('reading graph...', end=' ', flush=True)
    graph, _, _ = CSRCGraph.read_graph(args.graph)
    graph.pin_memory()
    print('Done!')

    # partition = VertexPartition(graph, num_partitions=1)
    partition = GeminiPartition(graph, num_partitions=1, alpha=8 * graph.num_vertices - 1)
    # strategy = MultiGPUOncePartition(partition, max_subgraphs_in_gpu=2)
    pr = PageRank(graph)
    # strategy = SimpleStrategy(partition, pr)
    # strategy = MultiGPUStrategyByNCCL(pr, partition, device_num=2)
    # strategy = MultiGPUStrategyByCupyAndNCCL(pr, partition, device_num=2)
    strategy = SimpleStrategy(pr)

    if args.cuda:
        # 计算策略中 移至对应设备
        pr.to('cuda')
        graph.to('cuda')
        print('use cuda')

    
    t1 = time.time()
    Tracer = VizTracer()
    Tracer.start()
    v_data, _ = pr.compute(strategy)
    x = v_data[0]
    Tracer.stop()
    t2 = time.time()
    Tracer.save()
    # print(v_data)
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    # with open(args.output, 'w') as f:
    #     for i in range(len(v_data[:, 0])):
    #         f.write(str(v_data[i, 0].item()) + '\n')


if __name__ == '__main__':
    main()
