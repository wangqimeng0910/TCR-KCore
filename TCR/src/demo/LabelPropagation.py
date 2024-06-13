"""
Label Propagation for semi-supervised learning. Assuming undirected graphs.
"""
import sys
sys.path.append('/home/lames/TCRGraph/')
import logging
from src.framework.strategy.SimpleStrategy import SimpleStrategy
from src.framework.GASProgram import GASProgram
from src.type.CSRGraph import CSRGraph
import torch
import argparse
import time
import random
from src.framework.helper import batched_csr_selection


class LabelPropagation(GASProgram):
    def __init__(self, graph: CSRGraph, vertex_data_type=torch.int32, edge_data_type=None, vertex_data=None, num_iter=50):
        # vertex_data: [num_vertices]. Each vertex has a label. By default each vertex has one.
        if vertex_data is None:
            vertex_data = torch.arange(0, graph.num_vertices, dtype=vertex_data_type)
        edge_data = None
        self.ITER_THRESHOLD = num_iter
        
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data)
        
    def gather(self, vertices, nbrs, edges, ptr):
        return self.vertex_data[nbrs], ptr
    
    def sum(self, gathered_data, ptr):
        result = torch.zeros(len(ptr) - 1, dtype=self.vertex_data_type)
        # get the most common label. If no mode, choose a random one.
        for i in range(len(ptr) - 1):
            data = gathered_data[ptr[i]:ptr[i+1]]
            indexes = torch.randperm(data.shape[0])
            data = data[indexes]
            if not data.cpu:
                data = data.cpu()
            label, _ = torch.mode(data)
            result[i] = label
        return result

    def apply(self, vertices, gathered_sum):
        return gathered_sum, torch.ones(len(vertices), dtype=torch.bool)
    
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        changed = apply_data != self.vertex_data[vertices]
        if self.nbr_update_freq > 0:
            if changed.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD:
                # get neighbors of changed
                starts, ends = ptr[changed], ptr[changed + 1]
                result, ptr = batched_csr_selection(starts, ends)
                all_neighbors = nbrs[result]
                self.activate(all_neighbors)
        else:
            if changed.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD:
                self.not_change_activated_next_iter()
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
    parser.add_argument('--labels', type=str, help='initial labels for vertices (By default starting from 0)')
    parser.add_argument('--maxsteps', type=int, help='max steps in iteration', default=3)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph = CSRGraph.read_graph(args.graph, split=None)
    graph = graph[0]
    labels = None
    if args.labels is not None:
        with open(args.labels, 'r') as f:
            labels = [int(line.strip()) for line in f]
        labels = torch.Tensor(labels, dtype=torch.int32)
    print('Done!')
    
    lpa = LabelPropagation(num_iter=args.maxsteps, graph=graph, vertex_data=labels)
    strategy = SimpleStrategy(lpa)
    t1 = time.time()
    lpa.compute(strategy)
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    with open(args.output, 'w') as f:
        for i in range(len(lpa.vertex_data[:])):
            f.write(str(lpa.vertex_data[i].item()) + '\n')

if __name__ == '__main__':
    main()
