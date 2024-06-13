"""
Implementation of PageRank algorithm using the GAS framework.
"""

import sys
import torch
import argparse
import time
import logging
from torch_scatter import segment_csr

sys.path.append('/home/lames/TCRGraph/')
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.framework.helper import batched_csr_selection
# from src.framework.strategy.PartitionStrategy import PartitionStrategy
# from src.framework.strategy.MultiGPUStrategy import MultiGPUStrategy
from src.framework.strategy.SimpleStrategy import SimpleStrategy
# from src.framework.strategy.MultiGPUOncePartition import MultiGPUOncePartition
# from src.framework.partition.VertexPartition import VertexPartition
from src.framework.partition.GeminiPartition import GeminiPartition

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class PageRank(GASProgram):
    def __init__(self, graph: CSRCGraph, vertex_data_type=torch.float32, edge_data_type=torch.float32, num_iter=50,
                 **kwargs):
        # vertex_data: [num_vertices, 2]. rank and delta
        vertex_data = torch.ones((graph.num_vertices, 2), dtype=vertex_data_type)
        # vertex_data /= graph.num_vertices
        self.UPDATE_THRESHOLD = 0.001
        self.ITER_THRESHOLD = num_iter
        self.out_degree = None
        super().__init__(graph, vertex_data=vertex_data, vertex_data_type=vertex_data_type,
                         edge_data_type=edge_data_type, **kwargs)

    def gather(self, vertices, nbrs, edges, ptr):
        if self.nbr_update_freq == 0:
            if self.out_degree is None:
                self.out_degree = 1 / self.graph.out_degree(nbrs).to(self.device)
        else:
            self.out_degree = 1 / self.graph.out_degree(nbrs).to(self.device)
        # print('data: {} out_degree: {}'.format(self.vertex_data[nbrs, 0], self.out_degree))
        data = self.vertex_data[nbrs, 0] * self.out_degree
        return data, ptr

    def sum(self, gathered_data, ptr):
        return segment_csr(gathered_data, ptr, reduce='sum')

    def apply(self, vertices, gathered_sum):
        rnew = 0.15 + 0.85 * gathered_sum
        delta = torch.abs(rnew - self.vertex_data[vertices, 0])
        return torch.stack([rnew, delta], dim=-1), torch.ones(vertices.shape + (2,), dtype=torch.bool,
                                                              device=self.device)

    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        if self.nbr_update_freq > 0:
            delta = self.vertex_data[vertices, 1]
            selected = torch.where(delta > self.UPDATE_THRESHOLD)[0]
            if selected.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD - 1:
                # get neighbors of selected
                starts, ends = ptr[selected], ptr[selected + 1]
                result, ptr = batched_csr_selection(starts, ends)
                all_neighbors = nbrs[result]
                self.activate(all_neighbors)
        else:
            if self.curr_iter < self.ITER_THRESHOLD - 1:
                self.not_change_activated_next_iter()
        return None, None

    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            in_nbrs, ptr = self.graph.all_in_nbrs_csr()
        else:
            in_nbrs, ptr = self.graph.in_nbrs_csr(vertices)
        return in_nbrs, None, ptr

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
    graph, _, _ = CSRCGraph.read_graph(args.graph)
    graph.pin_memory()
    print('Done!')


    partition = GeminiPartition(graph, num_partitions=1, alpha=8 * graph.num_vertices - 1)
    pr = PageRank(graph)
    strategy = SimpleStrategy(pr)
    vertices = torch.LongTensor([2, 3])
    g_nbrs, g_ptr = pr.graph.in_nbrs_csr(vertices)
    print(g_nbrs)
    print(g_ptr)
    g_sum = segment_csr(g_nbrs, g_ptr, reduce='min')
    print(g_sum)

    data = torch.LongTensor([0, 0, 0])
    nbrs = torch.LongTensor([0, 1, 0, 1])
    data[nbrs] += 1
    print(data[nbrs])
    print(data)



    # if args.cuda:
    #     pr.to('cuda')
    #     print('use cuda')
    #
    # t1 = time.time()
    # v_data, _ = pr.compute(strategy)
    # t2 = time.time()
    #
    # print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # # output results
    # with open(args.output, 'w') as f:
    #     for i in range(len(v_data[:, 0])):
    #         f.write(str(v_data[i, 0].item()) + '\n')


if __name__ == '__main__':
    main()
