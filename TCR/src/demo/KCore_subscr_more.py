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

import psutil
# from viztracer import VizTracer

def get_subCSR(row_ptr, columns, Vertex_num, deleted_nodes):
    if isinstance(deleted_nodes, torch.Tensor) and deleted_nodes.is_cuda:
        deleted_nodes = deleted_nodes.cpu()

    # 将list转换为NumPy数组
    row_ptr = np.array(row_ptr)
    columns = np.array(columns)
    deleted_nodes = np.array(deleted_nodes)

    # 初始化新的Row_ptr和Col数组
    new_Row_ptr = [0]
    new_Columns = []

    # 获取当前时间作为开始时间
    start_time = time.time()
    total_memory = psutil.virtual_memory().total  # 获取总内存量
    # start_mem = psutil.Process().memory_info().rss  # 获取当前进程的内存使用量
    print_interval = 10  # 设置打印间隔，单位为秒

    # 遍历每个顶点
    for vertex in range(Vertex_num):
        current_time = time.time()
        if current_time - start_time > print_interval:
            current_mem_usage = psutil.Process().memory_info().rss
            mem_usage_percent = (current_mem_usage / total_memory) * 100
            print("Processing vertex {:.2%}, Memory usage: {:.2f} MB".format(
                vertex / Vertex_num, 
                mem_usage_percent)
            )
            start_time = current_time  # 重置开始时间为当前时间

        if not deleted_nodes[vertex]:
            col_start = row_ptr[vertex]
            col_end = row_ptr[vertex + 1]
            valid_neighbors = columns[col_start:col_end][~deleted_nodes[columns[col_start:col_end]]]
            new_Columns.extend(valid_neighbors)
            new_Row_ptr.append(new_Row_ptr[-1] + len(valid_neighbors))
        else:
            new_Row_ptr.append(new_Row_ptr[-1])

    # 计算度数作为Row_ptr的差值
    new_degree = np.diff(new_Row_ptr)        
    # new_degree = [new_Row_ptr[i+1] - new_Row_ptr[i] for i in range(len(new_Row_ptr)-1)]

    # 输出新的Row_ptr和Col
    return new_Row_ptr, new_Columns, new_degree


def multi_arange(start,count):
    arr_len = torch.sum(count)
    # building reset indices
    ri=torch.zeros(torch.numel(count),dtype=count.dtype,device=count.device)
    ri[1:]=torch.cumsum(count,dim=0)[:-1]
    #building incremental indices
    incr=torch.ones(arr_len,dtype=count.dtype,device=count.device)
    incr[ri]=start
    # correcting start indices for initial values
    incr[ri[1:]]+=1-(start[:-1]+count[:-1])
    return torch.cumsum(incr,dim=0)

def get_weight(K,mu,sigma_squared):
    weights=torch.exp(-((K-mu)**2)/(2*sigma_squared))
    return weights

 
class mutilGPU_kcore(Process):
    def __init__(self,rank,size,row_ptr,columns,vertex_num,vertex_degrees,k_values, **kwargs):
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
        self.k_values = k_values

        self.vertex_data = torch.zeros(vertex_num, dtype=torch.int64)



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
        self.row_ptr = self.row_ptr.to(self.device)
        self.columns = self.columns.to(torch.long).to(self.device)
        self.vertex_degrees=self.vertex_degrees.to(self.device)
        self.check()

        dist.barrier(group=group)
        # tracer = VizTracer()
        # tracer.start()
        t1 = time.time()
        # Analyse_file=open('/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/{}_ana.txt'.format(filename),'a')
        # if self.rank==1:
        #     time.sleep(3)
        #I columns  D vertex_degrees II row_ptr K vertex_data
        CD = self.vertex_degrees.clone()
        C =torch.arange(self.vertex_num,dtype=torch.long,device=self.device)
        k_partition=self.k_values[self.rank:self.rank+2]
        k_start=k_partition[0]
        k_end=k_partition[1]
        k=k_start
      
        t1 = time.time()
        B=C[(self.vertex_degrees[C]<k)&(self.vertex_degrees[C]>0)]
        while B.shape[0]>0:
            self.vertex_degrees[B]=0

            nbrs_indexs=multi_arange(self.row_ptr[B],CD[B])
            nbrs=self.columns[nbrs_indexs]
            nbrs=nbrs[self.vertex_degrees[nbrs]>0]
            nodes,num=torch.unique(nbrs,return_counts=True)
            self.vertex_degrees[nodes]-=num
            B=nodes[(self.vertex_degrees[nodes]<k)]
        t2 = time.time()
        C=C[self.vertex_degrees[C]>=k]
        B=C[(self.vertex_degrees[C]<=k)]
        #标识一半区间

        while k<k_end and C.shape[0]>0:

            while B.shape[0]>0:
                self.vertex_degrees[B]=0
                self.vertex_data[B]=k.to(torch.int64)
                m=multi_arange(self.row_ptr[B],CD[B])
                J=self.columns[m]
                H=J[self.vertex_degrees[J]>0]
                H,cnt=torch.unique(H,return_counts=True)
                self.vertex_degrees[H]-=cnt                
                B=H[(self.vertex_degrees[H]<=k)]
            # Analyse_file.write("{},{},{},{}\n".format(k,t4-t3,count,num))
            k=k+1
            C=C[self.vertex_degrees[C]>=k]
            B=C[self.vertex_degrees[C]==k]
        # dist.all_reduce(self.vertex_data)
        t3 = time.time()
        # tracer.stop()
        # tracer.save()
        print("rank:{},----max_kk{}".format(self.rank,torch.max(self.vertex_data)))
        logging.info('rank {} finishSubGraph and time {}'.format(self.rank, t2 - t1))
        logging.info('rank {} k-core and time {}'.format(self.rank, t3 - t2))
        logging.info('rank {} finish and time {}'.format(self.rank, t3 - t1))



if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    mp.set_start_method('spawn', force=True)
    filename='gsh-2015'
    print("---------------------------{}---------------------------".format(filename))
    file_path='/root/autodl-tmp/dataset/{}'.format(filename)
    graph=CSRGraph.read_graph_bin(file_path,is_long=True)
    logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))
    num =3

    t3=time.time()
    deleted_nodes=torch.load('/root/autodl-tmp/dataset/{}/{}_deleted_nodes.pt'.format(filename,filename))
    new_Row_ptr,new_Columns,new_degree=get_subCSR(graph.row_ptr,graph.columns,graph.num_vertices,deleted_nodes)
    t4=time.time()
    logging.info('get_subCSR:{}'.format(t4-t3))
    new_Row_ptr=torch.tensor(new_Row_ptr).to(torch.long)
    new_Columns=torch.tensor(new_Columns).to(torch.long)
    new_degree=torch.tensor(new_degree).to(torch.long)
    logging.info('read graph done')
    #-----------------------按节点进行切分，保证每个k值间隔的顶点数目相同----------------------------------------
    # sorted_values, sorted_indices = torch.sort(graph.out_degrees)
    # vertex_degrees_cumsum=torch.cumsum(sorted_values,dim=0)
    # k_partition_num=graph.num_vertices/num
    # k_partition_num=torch.tensor(k_partition_num,dtype=torch.int64)
    # k_partitions=torch.arange(0,graph.num_vertices+1,k_partition_num).to(torch.int64)
    # k_partitions[-1]-=1
    # partition=sorted_values[k_partitions]
    # print("------",partition)
    # # print("k_values:{}".format(k_values))
    #-------------------------权重k值切分，保证每个区间内k值权重和相同-------------------------------------------
    degrees,counts=torch.unique(graph.out_degrees,return_counts=True)

    # Analyse_file=open('/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/com-lj.ungraph_degree.txt','a')
    # for i in range(len(degrees)):
    #   Analyse_file.write("{},{}\n".format(degrees[i],counts[i]))
    cumsum_reverse = torch.flip(torch.cumsum(torch.flip(counts, [0]), dim=0), [0])
    sum=degrees*counts
    avg_deg=torch.sum(sum)/graph.num_vertices
    middle_k=torch.searchsorted(torch.cumsum(counts,dim=0),graph.num_vertices/2)+1
    indexs=torch.where(cumsum_reverse>=degrees)[0]
    max_k=torch.max(degrees[indexs])
    print("-----------max_deg:",torch.max(degrees))
    print("-----------avg_deg:",avg_deg)
    print("-----------middle_k:",middle_k)
    print("-----------max_k:",max_k)    
    # gap=(middle_k-1)*100/max_k+avg_deg/middle_k
    # print("----------gap:",gap)
    # dev_factor=gap/10+1
    # if avg_deg-middle_k>middle_k-1:
    #     dev_factor=1.5+gap/10
    # else:
    #     dev_factor=1.5-gap/10
    # print("-----------dev_factor:",dev_factor)
    # dev_factor-=0.9
    # p=30
    # Analyse_file=open("'/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/com-lj.ungraph_{}".format(p))
    # Analyse_file.write("Hello, World!")
    # avg_k=0.00246*avg_deg+0.0249*middle_k+0.00112*max_k
    # avg_k=0-0.0000000459*avg_deg**2+0.0000154*middle_k**2+0.00000867*max_k**2-0.00039*avg_deg-0.00279*middle_k+0.00557*max_k-1.38
    # avg_k=5.11*avg_deg-6.24*middle_k-0.032*max_k+25
    # avg_k=5.11*avg_deg-6.24*middle_k-0.032*max_k+700
    avg_k=5.11*avg_deg-6.24*middle_k-0.032*max_k+torch.max(degrees)/4000
    K_arange=torch.arange(1,max_k+1)
    K_weights=get_weight(K_arange,avg_k,max_k).to(dtype=torch.float64)
    W_all=torch.cumsum(K_weights,dim=0)[-1]
    W_avg=W_all/num
    # P=torch.arange(0,W_all+W_avg,W_avg)
    P=torch.arange(0,W_all+1,W_avg)
    # P=torch.arange(0,W_all*dev_factor+1,W_avg*dev_factor)
    P[-1]+=1
    cum=torch.cumsum(K_weights,dim=0)
    cum[-1]+=1
    Index=torch.searchsorted(cum,P)
    partition=Index+1
    print("k_max:",max_k)
    print("------",partition)
   
    processes = []
    for i in range(num):
        # row_ptr = row_ptrs[i]
        # column = columns[i]
        # vertex_indices = vertex_begin_idx[i:i+2]
        # print('rank {} vertex length {}'.format(i, vertex_indices[1] - vertex_indices[0]))
        p = mutilGPU_kcore(i,num,new_Row_ptr,new_Columns,graph.num_vertices,new_degree,partition)
        # p = mutilGPU_kcore(i,num,graph.row_ptr,graph.columns,graph.num_vertices,graph.out_degrees,k_values)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()