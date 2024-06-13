import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import time
import sys
sys.path.append('/root/autodl-tmp/TCRGraph-tcr_gas')
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.type.Subgraph import Subgraph
from src.type.Graph import Graph
import logging 
from torch_scatter import segment_csr
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
import torch.multiprocessing as mp

import numpy as np

# import viztracer 

k_stop=300
filename='uk-2007'

def multi_arange(start,count):
    # print(start.dtype,count.dtype)
    arr_len = torch.sum(count)
    # print("arr_len:",arr_len.dtype)
    # building reset indices
    ri=torch.zeros(torch.numel(count),dtype=count.dtype,device=count.device)
    ri[1:]=torch.cumsum(count,dim=0)[:-1]
    #building incremental indices
    incr=torch.ones(arr_len,dtype=count.dtype,device=count.device)
    incr[ri]=start
    # correcting start indices for initial values
    incr[ri[1:]]+=1-(start[:-1]+count[:-1])
    return torch.cumsum(incr,dim=0)

# 点均衡，即切割后每个区间点相同
def get_partition2(graph: Graph, partition_num: int):
    row_ptr = None
    columns = None
    degrees = None
    if isinstance(graph, CSRCGraph):
        degrees = graph.csc.csr.out_degrees
        row_ptr = graph.csc.csr.row_ptr
        columns = graph.csc.csr.columns
    elif isinstance(graph, CSRGraph):
        degrees = graph.out_degrees
        row_ptr = graph.row_ptr
        columns = graph.columns
    row_ptr = row_ptr.to(torch.long)
    degrees = degrees.to(torch.long)
    partition_size = graph.num_vertices // partition_num 
    vertex_begin_idx = []
    for i in range(partition_num):
        vertex_begin_idx.append(i*partition_size)
    vertex_begin_idx.append(graph.num_vertices)
    vertex_begin_idx = torch.tensor(vertex_begin_idx, dtype=torch.int64)
    parts_row_ptr = []
    parts_columns = []

    for i in range(partition_num):
        part_degrees = degrees[vertex_begin_idx[i]:vertex_begin_idx[i + 1]]
        part_row_ptr = torch.cumsum(part_degrees, dim=0)
        part_row_ptr = torch.cat([torch.tensor([0], dtype=torch.int64), part_row_ptr])
        col_begin=row_ptr[vertex_begin_idx[i]]
        col_end=row_ptr[vertex_begin_idx[i+1]]
        part_column = columns[col_begin : col_end]
        part_row_ptr = part_row_ptr.to(torch.long)
        part_column = part_column.to(torch.int32)
        logging.info('part {} row_ptr {} columns {}'.format(i, part_row_ptr.numel(), part_column.numel()))
        parts_row_ptr.append(part_row_ptr)
        parts_columns.append(part_column)
  
    return parts_row_ptr, parts_columns, vertex_begin_idx


# 边均衡，即切割后每个区间边相同
def get_partition(graph: Graph, partition_num: int):
    row_ptr = None
    columns = None
    degrees = None

    if isinstance(graph, CSRCGraph):
        degrees = graph.csc.csr.out_degrees
        row_ptr = graph.csc.csr.row_ptr
        columns = graph.csc.csr.columns

    elif isinstance(graph, CSRGraph):
        degrees = graph.out_degrees
        row_ptr = graph.row_ptr
        columns = graph.columns

    
    # print(degrees.dtype)
    row_ptr = row_ptr.to(torch.long)
    degrees = degrees.to(torch.long)
    degrees_ptr = row_ptr[1:] - row_ptr[:-1]
    mask = degrees_ptr == degrees
    
    degrees_sum = torch.cumsum(degrees, dim=0, dtype=torch.int64)
    all_sum = degrees_sum[-1]


    # print('all_sum {}'.format(all_sum))
    print('row_ptr {}'.format(row_ptr[-1]))
    partition_size = all_sum // partition_num
    target_sizes = [i * partition_size for i in range(1, partition_num)]
    target_sizes = torch.tensor(target_sizes, dtype=torch.int64)
    indexs = torch.searchsorted(degrees_sum, target_sizes)

    vertex_begin_idx = [0]
    for i in range(partition_num - 1):
        vertex_begin_idx.append(indexs[i] + 1)
    vertex_begin_idx.append(graph.num_vertices)
    vertex_begin_idx = torch.tensor(vertex_begin_idx, dtype=torch.int64)

    parts_row_ptr = []
    parts_columns = []

    for i in range(partition_num):
        part_degrees = degrees[vertex_begin_idx[i]:vertex_begin_idx[i + 1]]
        part_row_ptr = torch.cumsum(part_degrees, dim=0)
        part_row_ptr = torch.cat([torch.tensor([0], dtype=torch.int64), part_row_ptr])
        
        col_begin = row_ptr[vertex_begin_idx[i]]
        col_end = row_ptr[vertex_begin_idx[i + 1]]
        part_columns = columns[col_begin : col_end]
        
        part_row_ptr = part_row_ptr.to(torch.long)
        part_columns = part_columns.to(torch.int32)
        logging.info('part {} row_ptr {} columns {}'.format(i, part_row_ptr.numel(), part_columns.numel()))
        parts_row_ptr.append(part_row_ptr)
        parts_columns.append(part_columns)
    
    return parts_row_ptr, parts_columns, vertex_begin_idx

class mutilGPU_kcore(Process):
    def __init__(self,rank,size,row_ptr,columns,vertex_num,vertex_degrees,vertex_indices, **kwargs):
        super().__init__(**kwargs)
        self.rank = rank
        self.size = size
        self.device = f'cuda:{self.rank}'
        logging.info('rank {} device {}'.format(self.rank, self.device))
        self.row_ptr = row_ptr
        self.columns = columns
        # num of vertex
        self.vertex_num = vertex_num
        # degress of degrees
        self.vertex_degrees = vertex_degrees
        # vertex indices to indicate the begin and end of the vertex owned to this process
        self.vertex_indices = vertex_indices
        self.vertex_data = torch.zeros(self.vertex_indices[1]-self.vertex_indices[0], dtype=torch.int64)

    def check(self):
        # memory = self.row_ptr.numel() * 4 + self.columns.numel() * 12 + self.vertex_data.numel() * 4
        # memory = memory / 1024 / 1024 / 1024
        # logging.info('rank {} memory {} GB'.format(self.rank, memory))
        allc = torch.cuda.memory_allocated(self.device) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(self.device) / 1024 ** 3
        logging.info('rank {} allocated {} GB, reserved {} GB'.format(self.rank, allc, reserved))
    
    def init_processes(self, backend='nccl'):
        """ Initialize the distributed environment. """
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '30036'
        dist.init_process_group(backend, rank=self.rank, world_size=self.size)
        logging.info('rank {} init process done'.format(self.rank))



    def run(self):
        torch.cuda.empty_cache()
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        self.init_processes()
        logging.info('rank {} start'.format(self.rank))
        group = dist.new_group([i for i in range(self.size)])

        # 消除初始化环境的影响
        self.vertex_data = self.vertex_data.to(self.device)
        logging.info('rank {} vertex data {}'.format(self.rank, self.vertex_data.numel() * 4 / 1024 ** 3))
        self.row_ptr = self.row_ptr.to(torch.long).to(self.device)
        self.columns = self.columns.to(torch.long).to(self.device)
        self.vertex_degrees=self.vertex_degrees.to(torch.long).to(self.device)
        self.check()

        dist.barrier(group=group)
        # tracer = viztracer.VizTracer()
        # tracer.start()

        #I columns  D vertex_degrees II row_ptr K vertex_data
        CD = self.vertex_degrees.clone()
        Vertex_list =torch.arange(self.vertex_indices[0],self.vertex_indices[1],dtype=torch.long,device=self.device)
        k=1
        B=Vertex_list[self.vertex_degrees[Vertex_list]<=k]
        #标识一半区间
        flag =torch.zeros(self.vertex_num,dtype=bool,device=self.device)
        flag[Vertex_list]=True
        t1 = time.time()
        #控制全局迭代同步
        boolflag_local_C=torch.zeros(1,dtype=torch.int64,device=self.device)
        boolflag_all_C=torch.zeros(1,dtype=torch.int64,device=self.device)
        while boolflag_all_C==0:
            #控制每轮迭代中同步
            boolflag_local_B=torch.zeros(1,dtype=torch.int64,device=self.device)
            boolflag_all_B=torch.zeros(1,dtype=torch.int64,device=self.device)
            while boolflag_all_B==0:
                self.vertex_degrees[B]=0
                self.vertex_data[B-self.vertex_indices[0]]=k
                self.vertex_degrees*=flag
                #同步全局度数
                dist.all_reduce(self.vertex_degrees,op=dist.ReduceOp.SUM,group=group)
                m=multi_arange(self.row_ptr[B-self.vertex_indices[0]],CD[B])
                #J是全局的邻居
                J=self.columns[m]
                H=J[self.vertex_degrees[J]>0]
                H,cnt=torch.unique(H,return_counts=True)
                tag=torch.zeros(self.vertex_num,dtype=torch.int64,device=self.device)
                tag[H]=cnt
                dist.all_reduce(tag,op=dist.ReduceOp.SUM,group=group)
                self.vertex_degrees-=tag
                #tag是所有顶点的变化值，tag[Vertex_list]是Vertex_list范围内对应顶点的变化值，tag[C]>0是C范围内对应顶点的变化值大于1的点
                #H 本区间内变化的数据
                H=Vertex_list[tag[Vertex_list]>0]
                B=H[self.vertex_degrees[H]<=k] 
                if B.shape[0]==0:
                    boolflag_local_B=torch.ones(1,dtype=torch.int64,device=self.device)
                else:
                    boolflag_local_B=torch.zeros(1,dtype=torch.int64,device=self.device)
                dist.all_reduce(boolflag_local_B,op=dist.ReduceOp.PRODUCT,group=group)
                boolflag_all_B=boolflag_local_B
                if B.shape[0]==0:
                    boolflag_local_B=torch.ones(1,dtype=torch.int64,device=self.device)
            # dist.barrier(group=group)
            k=k+1
            Vertex_list=Vertex_list[self.vertex_degrees[Vertex_list]>=k]
            B=Vertex_list[self.vertex_degrees[Vertex_list]==k]
            #本卡迭代完毕或到底阈值
            if Vertex_list.shape[0]==0 or k==k_stop:
                boolflag_local_C=torch.ones(1,dtype=torch.int64,device=self.device)
            dist.all_reduce(boolflag_local_C,op=dist.ReduceOp.PRODUCT,group=group)
            boolflag_all_C=boolflag_local_C
            #本卡迭代完毕或到底阈值
            if Vertex_list.shape[0]==0 or k==k_stop:
                boolflag_local_C=torch.ones(1,dtype=torch.int64,device=self.device)
            # print("-----------rank:{},k={},end---------------".format(self.rank,k))
        t2 = time.time()
        print("rank:{},----max_kk{}".format(self.rank,torch.max(self.vertex_data)))
        logging.info('rank {} finish and time {}'.format(self.rank, t2 - t1))
        #收集处理过的顶点，用于缩减子图
        del_node=torch.where(self.vertex_data>0)[0]
        deleted_nodes=torch.zeros(self.vertex_num,dtype=bool,device=self.device)
        del_node=self.vertex_indices[0]+del_node
        deleted_nodes[del_node]=True
        dist.all_reduce(deleted_nodes,op=dist.ReduceOp.SUM,group=group)
        logging.info('rank {} deleted_nodese {}'.format(self.rank, deleted_nodes))
        #保存数据
        torch.save(deleted_nodes,'/root/autodl-tmp/dataset/{}/{}_deleted_nodes.pt'.format(filename,filename))
        

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    mp.set_start_method('spawn', force=True)
    path = '/root/autodl-tmp/dataset/{}'.format(filename)
    graph=CSRGraph.read_graph_bin(path,is_long=True)
    
    logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))
    #num表示卡的数目
    num =4
    degrees,counts=torch.unique(graph.out_degrees,return_counts=True)

    # Analyse_file=open('/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/com-lj.ungraph_degree.txt','a')
    # for i in range(len(degrees)):
    #   Analyse_file.write("{},{}\n".format(degrees[i],counts[i]))
    cumsum_reverse = torch.flip(torch.cumsum(torch.flip(counts, [0]), dim=0), [0])
    sum=degrees*counts
    avg_deg=torch.sum(sum)/graph.num_vertices
    middle_k=torch.searchsorted(torch.cumsum(counts,dim=0),graph.num_vertices/2)+1
    indexs=torch.where(cumsum_reverse>=degrees)[0]
    max_k=torch.max(degrees[indexs],dim=0)
    #图的特性
    print("-----------max_deg:",torch.max(degrees))
    print("-----------avg_deg:",avg_deg)
    print("-----------middle_k:",middle_k)
    print("-----------max_k:",max_k)

    #TCR下的数据切分
    row_ptrs, columns, vertex_begin_idx = get_partition(graph, num)
    vertex_num = graph.num_vertices
    vertex_degrees = graph.out_degrees

    processes = []
    for i in range(num):
        row_ptr = row_ptrs[i]
        column = columns[i]
        vertex_indices = vertex_begin_idx[i:i+2]
        print('rank {} vertex length {}'.format(i, vertex_indices[1] - vertex_indices[0]))
        p = mutilGPU_kcore(i,num,row_ptr,column,vertex_num,vertex_degrees,vertex_indices)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()