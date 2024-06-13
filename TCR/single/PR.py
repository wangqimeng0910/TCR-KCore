import torch
import sys
sys.path.append('/root/Project/TCRGraph/')
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph
import argparse
from torch_scatter import segment_csr, scatter_add, segment_coo
import time

class PageRank(GASProgram):
    def __init__(self, graph: CSRCGraph, vertex_data_type=torch.float32, edge_data_type=torch.float32, num_iter=25,
                 **kwargs):
        # vertex_data: [num_vertices, 2]. rank and delta
        # vertex_data = torch.ones((graph.num_vertices, 2), dtype=vertex_data_type)
        # vertex_data /= graph.num_vertices
        self.UPDATE_THRESHOLD = 0.001
        self.ITER_THRESHOLD = num_iter
        self.out_degree = None
        # self.index = torch.arange(graph.num_vertices, dtype=torch.int64, device=graph.device).\
        #     repeat_interleave(torch.diff(graph.csr.row_ptr)).cuda()
        # self.out = torch.zeros(graph.num_vertices, dtype=torch.float32).cuda()
        vertex_data = torch.ones(graph.num_vertices, dtype=vertex_data_type).to(graph.device)
        self.vertex_delta = torch.ones(graph.num_vertices, dtype=vertex_data_type).to(graph.device)
        super().__init__(graph, vertex_data=vertex_data, vertex_data_type=vertex_data_type,
                         edge_data_type=edge_data_type, **kwargs)

    def gather(self, vertices, nbrs, edges, ptr):
        if self.nbr_update_freq == 0:
            if self.out_degree is None:
                self.out_degrees = 1 / self.graph.csr.out_degrees.to(self.device) * 0.85
                self.out_degree = self.out_degrees[nbrs]
        else:
            self.out_degree = 1 / self.graph.out_degree(nbrs).to(self.device)
        # print('data: {} out_degree: {}'.format(self.vertex_data[nbrs, 0], self.out_degree))
        # data = self.vertex_data[nbrs, 0] * self.out_degree
        data = self.vertex_data * self.out_degrees
        data = data[nbrs]
        # data = self.vertex_data[nbrs, 0] * self.out_degree
        return data, ptr

    def sum(self, gathered_data, ptr):
        # import logging
        # logging.info('gathered_data: {}'.format(gathered_data.device))
        # logging.info('index: {}'.format(self.index.device))
        # logging.info('out: {}'.format(self.out.device))
        # scatter_add(gathered_data, self.index, out=self.out)
        # return self.out
        # return segment_csr(gathered_data, ptr, reduce='sum')
        # return segment_coo(gathered_data, self.index, reduce='sum')
        return segment_csr(gathered_data, ptr, reduce='sum')
    def apply(self, vertices, gathered_sum):
        """
        Returns:
            apply_data: 
            apply_mask: 
        """
        # logging.info('gathered_sum: {}'.format(gathered_sum))
        # logging.info('vertices: {}'.format(vertices))
        
        rnew = gathered_sum + 0.15
        delta = torch.abs(rnew - self.vertex_data)

        # self.vertex_data[:, 0] = rnew
        # self.vertex_data[:, 1] = delta
        self.vertex_data = rnew
        self.vertex_delta = delta
        # return torch.stack([rnew, delta], dim=-1), torch.ones(vertices.shape + (2,), dtype=torch.bool,
        #                                                       device=self.device)
        return None, None
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        # if self.nbr_update_freq > 0:
        #     delta = self.vertex_data[vertices, 1]
        #     selected = torch.where(delta > self.UPDATE_THRESHOLD)[0]
        #     if selected.shape[0] > 0 and self.curr_iter < self.ITER_THRESHOLD - 1:
        #         # get neighbors of selected
        #         starts, ends = ptr[selected], ptr[selected + 1]
        #         result, ptr = batched_csr_selection(starts, ends)
        #         all_neighbors = nbrs[result]
        #         self.activate(all_neighbors)
        # else:
        #     if self.curr_iter < self.ITER_THRESHOLD - 1:
        #         self.not_change_activated_next_iter()

        if self.curr_iter < self.ITER_THRESHOLD - 1:
            if self.curr_iter != 0 and self.curr_iter % 10 == 0:
                # selected = torch.where(self.vertex_data[:, 1] > self.UPDATE_THRESHOLD)[0]
                # logging.info('selected: {}'.format(selected.shape))
            #     if selected.shape[0] == 0:
            #         self.is_quit = True
                pass 
        else:
            self.is_quit = True
        # if self.curr_iter < self.ITER_THRESHOLD - 1:
        #     self.not_change_activated_next_iter()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    args = parser.parse_args()
    device = 'cpu'
    graph, _, _ = CSRCGraph.read_graph(args.graph)
    if device == 'cuda':
        graph.pin_memory()
    graph.to(device)
    num_iter = 25
    # num_vertices = graph.num_vertices
    # vertex_data = torch.ones(num_vertices, dtype=torch.float32, device='cuda:0')
    # vertex_delta = torch.ones(num_vertices, dtype=torch.float32, device='cuda:0')
    # degree = None
    # row_ptr = graph.csr.row_ptr
    # columns = graph.csr.columns 
    # # print(vertex_data[0,0])
    # t1 = time.time()
    # for i in range(num_iter):
    #     if degree == None:
    #         degree = 1 / graph.csr.out_degrees
    #     g_data = vertex_data * degree
    #     s_data = segment_csr(g_data, graph.csr.row_ptr, reduce='sum')
    #     rnew = 0.15 + 0.85 * s_data
    #     vertex_delta = torch.abs(vertex_data - rnew)
    #     vertex_data = rnew
    #     del g_data
    #     del s_data
    #     del rnew
    # # torch.cuda.empty_cache()
    # # ans = vertex_data[0,0]
    # # print(ans)
    # print(vertex_data[0])
    # t2 = time.time()
    # print(t2 - t1)

    activated_list = torch.arange(graph.num_vertices, dtype=torch.int32, device=device)
    pr = PageRank(graph)
    pr.to(device)

    from viztracer import VizTracer
    tracer = VizTracer()
    t1 = time.time()
    tracer.start()
    for i in range(13):
        if i == 0:
            pr.g_nbrs, pr.g_edges, pr.g_ptr = pr.gather_nbrs(activated_list)
        gd, gd_ptr = pr.gather(activated_list, pr.g_nbrs, pr.g_edges, pr.g_ptr)
        gsum = pr.sum(gd, gd_ptr)
        pr.apply(activated_list, gsum)
    tracer.stop()
    tracer.save()
    # torch.cuda.synchronize()
    t2 = time.time()

    print('all time', t2 - t1)