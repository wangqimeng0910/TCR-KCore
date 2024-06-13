"""
利用MPI实现单机多卡Demo, PageRank算法
"""
from mpi4py import MPI
import argparse
import sys
import torch
import logging
import time
import numpy as np
sys.path.append('/home/lames/TCRGraph01/')
from src.type.CSRCGraph import CSRCGraph
from src.type.Subgraph import Subgraph
from src.framework.partition.GeminiPartition import GeminiPartition
from src.framework.partition.VertexPartition import VertexPartition
from torch_scatter import segment_csr
from viztracer import VizTracer
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)


if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    # parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    # parser.add_argument('--cuda', action='store_true', help='use cuda')
    args = parser.parse_args()
    x = torch.Tensor([0])
    x.to(f'cuda:{rank}')
    # rank = 0 进程上进行图读取和图切割, 并将切割后的子图分发给其他进程（包括自己）
    if rank == 0:
        logging.info('process {} is reading graph...'.format(rank))
        graph, _, _ = CSRCGraph.read_graph(args.graph)
        graph.pin_memory()
        logging.info('process {} Done!'.format(rank))
        num_partitions = comm.Get_size()
        partition = GeminiPartition(graph, num_partitions=num_partitions, alpha=8 * num_partitions - 8)
        # partition = VertexPartition(graph, num_partitions=num_partitions)
        # 生成切割子图
        tbegin = MPI.Wtime()
        tp1 = MPI.Wtime()
        partitions = partition.generate_partitions()
        tp2 = MPI.Wtime()
        logging.info('process {} generate_partitions time {}'.format(rank, tp2 - tp1))
        subgraph = []
        for (p, i), v in partitions:
            orig_to_sub_vertices = torch.zeros_like(graph.vertices)
            orig_to_sub_vertices[v] = torch.arange(v.numel())
            new_subgraph = Subgraph(p, v, i, orig_to_sub_vertices)
            # logging.info('v {}'.format(v))
            subgraph.append(new_subgraph)
        num_vertices = graph.num_vertices
        out_degree = graph.out_degree(graph.vertices)
        t2 = MPI.Wtime()
        logging.info('process {} read graph time {}'.format(rank, t2 - tbegin))
    else:
        subgraph = None
        num_vertices = None
        out_degree = None
        tbegin = time.time()
        
    all_time0 = MPI.Wtime() 
    logging.info('process {} all_time0 {}'.format(rank, all_time0))
    tracer = VizTracer()
    tracer.start()
    # scatter subgraphs
    subgraph = comm.scatter(subgraph, root=0)
    # logging.info('process {} is computing... {}'.format(rank, subgraph.sub_vertices))
    # broadcast num_vertices to generate vertex_data
    num_vertices = comm.bcast(num_vertices, root=0)
    # logging.info('process {} vertex_num {}'.format(rank, num_vertices))
    # broadcast out_degree 
    out_degree = comm.bcast(out_degree, root=0)

    all_time1 = MPI.Wtime()
    logging.info('process {} all_time1 {}'.format(rank, all_time1))
    # vertex_data = torch.ones((num_vertices, 2), dtype=torch.float32)
    vertex_data_np = np.ones((num_vertices, 2), dtype=np.float32)
    a = time.time()
    logging.info('process {} a {}'.format(rank, a))
    changed_vertex_mask = torch.zeros(num_vertices, dtype=torch.bool)
    b = time.time()
    logging.info('process {} b {}'.format(rank, b))
    # logging.info('process {} vertex_data {}'.format(rank, vertex_data))
    changed_vertex_mask[subgraph.sub_vertices] = True
    logging.info('process {} num_vertices {}'.format(rank, subgraph.sub_vertices.shape[0]))
    c = time.time()
    logging.info('process {} c {}'.format(rank, c))
    changed_vertex_mask = torch.stack([changed_vertex_mask, changed_vertex_mask], dim=-1)
    d = time.time()
    logging.info('process {} d {}'.format(rank, d))
    all_time2 = MPI.Wtime()
    logging.info('process {} all_time2 {}'.format(rank, all_time2))
    logging.info('process {} prepare_time {}'.format(rank, all_time2 - all_time0))
    # do computation
    compute_time = 0
    communation_time = 0  

    # to GPU
    a = time.time()
    logging.info('process {} begin_time {}'.format(rank, a))
    changed_vertex_mask = changed_vertex_mask.to(f'cuda:{rank}', non_blocking=True)
    subgraph.to(f'cuda:{rank}', non_blocking=True)
    out_degree = out_degree.to(f'cuda:{rank}', non_blocking=True)
    b = time.time()
    logging.info('process {} to GPU time {}'.format(rank, b - a))

    num_iters = 20
    for i in range(num_iters):
        vertex_data = torch.from_numpy(vertex_data_np).to(f'cuda:{rank}', non_blocking=True)
        logging.info('process {} iter {}'.format(rank, i))
        t1 = MPI.Wtime()
        logging.info('process {} compute begin time {}'.format(rank, t1))
        # gather
        g_nbrs, g_ptr = subgraph.all_in_nbrs_csr()
        gather_data = 1 / out_degree[g_nbrs] * vertex_data[g_nbrs, 0]
        # sum
        gsum = segment_csr(gather_data, g_ptr, reduce='sum')
        # apply
        rnew = 0.15 + 0.85 * gsum
        delta = torch.abs(rnew - vertex_data[subgraph.sub_vertices, 0])
        # logging.info('process {} rnew {}'.format(rank, rnew))
        apply_data = torch.stack([rnew, delta], dim=-1)
        # logging.info('process {} vertex {} apply_data {}'.format(rank, subgraph.sub_vertices, rnew))
        vertex_data[subgraph.sub_vertices] = apply_data
        vertex_data_local = torch.where(changed_vertex_mask, vertex_data, torch.zeros((num_vertices, 2), dtype=torch.float32, device=f'cuda:{rank}'))
        # logging.info('process {} vertex_data_local {}'.format(rank, vertex_data_local))
        vertex_data_local_np = vertex_data_local.cpu().numpy()
        t2 = MPI.Wtime()
        logging.info('process {} compute time one iter {}'.format(rank, t2 - t1))
        logging.info('process {} compute end time {}'.format(rank, t2))
        compute_time += t2 - t1
        # MPI Reduce vertex_data 
        # need transfer data from GPU to CPU for MPI Communication
        t1 = MPI.Wtime()
        logging.info('process {} reduce begin time {}'.format(rank, t1))
        comm.Allreduce(vertex_data_local_np, vertex_data_np, op=MPI.SUM)
        t2 = MPI.Wtime()
        logging.info('process {} reduce end time {}'.format(rank, t2))
        logging.info('process {} Allreduce time {}'.format(rank, t2 - t1))
        communation_time += t2 - t1
        # logging.info('process {} vertex_data {}'.format(rank, vertex_data))
        
    all_time1 = MPI.Wtime()
    tracer.stop()
    logging.info('process {} all_time {}'.format(rank, all_time1 - tbegin))
    logging.info('process {} compute_time {} communation_time {} all_time {}'.format(rank, compute_time, communation_time, 
                                                                                     compute_time + communation_time))
    # logging.info('process {} vertex_data {}'.format(rank, vertex_data))
    # print('process {} vertex_data {}'.format(rank, vertex_data_local))
