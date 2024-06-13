"""
Impletation of Triangle Counting Algorithm using TCRGraph
"""

import sys
import torch
import argparse
import time
import logging
from torch_scatter import segment_csr
# sys.path.append('/home/lames/TCRGraph/')
sys.path.append('/root/Project/TCRGraph/')
from src.type.Graph import Graph
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.framework.helper import batched_csr_selection
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.partition.GeminiPartition import GeminiPartition

logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


class TriangleCounting(GASProgram):
    def __init__(self, graph: Graph, vertex_data_type=torch.int32, edge_data_type=torch.float32, vertex_data=None, edge_data=None, start_from=None, nbr_update_freq=0):
        # store the number of triangles for every vertex
        self.vertex_data = torch.zeros((graph.num_vertices, 1), dtype=vertex_data_type)
        # compute the activated vertex (degree > 1)
        self.degree = graph.all_degree(graph.vertices)
        mask = torch.eq(self.degree, 1).logical_not()
        start_from = graph.vertices[mask]
        # compute adjacent matrix  remeber move to GPU  
        self.matrix = torch.sparse_csr_tensor(graph.row_ptr, graph.columns, torch.ones(len(graph.columns), dtype=torch.int32)).to_dense().to(torch.int16)
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data, start_from, nbr_update_freq)

    def gather(self, vertices, nbrs, edges, ptr):
        # to(torch.int16) 这里的数据类型可以优化, small is better
        gather_data = self.matrix[nbrs]
        return gather_data, ptr

    def sum(self, gathered_data, ptr):
        """
        segment_csr 
        """
        return segment_csr(gathered_data, ptr, reduce='sum')

    def apply(self, vertices, gathered_sum):
        """
        Returns:
            apply_data: 
            apply_mask: 
        """
        apply_mask = torch.ones(len(vertices), dtype=torch.bool, device=self.device)
        tmp_data = self.matrix[vertices]
        apply_data = torch.sum(tmp_data * gathered_sum, dim=1) // 2
        # logging.info('gathered_sum: {}'.format(gathered_sum))
        # logging.info('apply_data: {}'.format(apply_data))
        # logging.info('tmp_data: {}'.format(tmp_data))
        apply_data = apply_data.to(torch.int32)
        # logging.info('apply_data: {}'.format(apply_data.dtype))
        # logging.info('gather_sum: {}'.format(gathered_sum.dtype))
        return apply_data, apply_mask

    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        """
        no need to scatter 
        """
        return None, None

    def gather_nbrs(self, vertices):
        out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr

    def scatter_nbrs(self, vertices):
        """
        no need to scatter
        """
        # out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return None, None, None
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()

    print('reading graph...', end=' ', flush=True)
    graph, _ = CSRGraph.read_graph(args.graph)
    graph.pin_memory()
    print('Done!')

    # partition = VertexPartition(graph, num_partitions=1)
    partition = GeminiPartition(graph, num_partitions=1, alpha=8 * graph.num_vertices - 1)
    # strategy = MultiGPUOncePartition(partition, max_subgraphs_in_gpu=2)
    tc = TriangleCounting(graph, nbr_update_freq=1)
    # strategy = SimpleStrategy(partition, pr)
    strategy = SimpleStrategy(tc)

    if args.cuda:
        tc.to('cuda:0')
        tc.matrix = tc.matrix.to('cuda:0')
        print('use cuda')

    # from viztracer import VizTracer
    # Tracer = VizTracer()
    t1 = time.time()
    # Tracer.start()
    v_data, _ = tc.compute(strategy)
    # Tracer.stop()
    # Tracer.save()
    t2 = time.time()

    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    # with open(args.output, 'w') as f:
    #     for i in range(len(v_data[:, 0])):
    #         f.write(str(v_data[i, 0].item()) + '\n')


if __name__ == '__main__':
    main()