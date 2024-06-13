"""
Implementation of the k-Core algorithm. Undirected. Modeled after paper "Distributed k-Core Decomposition" (Montresor et al., 2011)
"""

from ..framework.GASProgram import GASProgram
from ..type.CSRGraph import CSRGraph
import torch
import argparse
import time
from ..framework.helper import batched_csr_selection

class KCore(GASProgram):
    def __init__(self, graph: CSRGraph, vertex_data_type=torch.int32, vertex_data=None, num_iter=100):
        # each vertex keeps an estimation of its core number. default to degrees
        if vertex_data is None:
            vertex_data = graph.out_degree(graph.vertices)
        self.ITER_THRESHOLD = num_iter
        # no edge_data needed
        super().__init__(graph, vertex_data=vertex_data, vertex_data_type=vertex_data_type)
        
    def gather(self, vertices, nbrs, edges, ptr):
        # collect estimations from neighbors
        return (self.vertex_data[vertices], self.vertex_data[nbrs]), ptr
    
    def sum(self, gathered_data, ptr):
        vertices_data, nbrs_data = gathered_data
        result = torch.zeros((len(ptr) - 1,), dtype=self.vertex_data_type, device=self.device)
        for i in range(len(ptr) - 1):
            curr_est = vertices_data[i].item()
            nbr_est = nbrs_data[ptr[i]:ptr[i+1]]
            new_est = self.compute_index(curr_est, nbr_est)
            result[i] = new_est
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
    
    @staticmethod
    def compute_index(curr_est, nbr_est):
        """
        Compute current estimation for specified vertex. The coreness of node u is the largest value k such that k has at least k neighbors that belong to a k-core or a larger core.
        
        :param int curr_est: core number estimation for the current node (u)
        :param List[int] curr_est: core number estimations for the neighbors of the current node (u)
        :return: new core number estimation for the current node (u)
        """
        count = [0 for _ in range(curr_est + 1)]
        for v_est in nbr_est:
            v_est = min(curr_est, v_est)
            count[v_est] += 1
        for i in range(curr_est - 1, 0, -1):
            count[i] += count[i + 1]
        new_est = curr_est
        while new_est > 1 and count[new_est] < new_est:
            new_est -= 1
        return new_est

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph, _ = CSRGraph.read_graph(args.graph, split=None)
    print('Done!')
    
    kcore = KCore(graph=graph)
    
    t1 = time.time()
    kcore.compute()
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    with open(args.output, 'w') as f:
        for i in range(len(kcore.vertex_data[:])):
            f.write(str(kcore.vertex_data[i].item()) + '\n')

if __name__ == '__main__':
    main()
