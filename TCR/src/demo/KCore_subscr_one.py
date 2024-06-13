import torch
import sys
sys.path.append('/root/autodl-tmp/TCRGraph-tcr_gas')
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph, CSRGraph
import argparse
from torch_scatter import segment_csr, scatter_add, segment_coo
import time
# from viztracer import VizTracer

import logging 
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)

import numpy as np
import psutil

DEBUG = False


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



def main():
    device = torch.device('cuda:0')
    filename='uk-2007'
    path = '/root/autodl-tmp/dataset/{}'.format(filename)
    graph=CSRGraph.read_graph_bin(path,is_long=True)
    logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))
    t3=time.time()
    deleted_nodes=torch.load('/root/autodl-tmp/dataset/{}/{}_deleted_nodes.pt'.format(filename,filename))
    new_Row_ptr,new_Columns,new_degree=get_subCSR(graph.row_ptr,graph.columns,graph.num_vertices,deleted_nodes)
    t4=time.time()
    logging.info('get_subCSR:{}'.format(t4-t3))
    new_Row_ptr=torch.tensor(new_Row_ptr)
    new_Columns=torch.tensor(new_Columns)
    new_degree=torch.tensor(new_degree)
    # logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))
    logging.info('read graph done')
    allc = torch.cuda.memory_allocated(device) / 1024 ** 3
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 3
    logging.info('allocated {} GB, reserved {} GB'.format(allc, reserved))

    I=new_Columns.to(torch.long).to(device)
    D=new_degree.to(torch.long).to(device)
    II=new_Row_ptr.to(torch.long).to(device)

    K=torch.zeros(graph.num_vertices, dtype=torch.long, device=device)
    # copy of D to remain unchanged
    CD=D.clone()
    C=torch.arange(graph.num_vertices,dtype=torch.long, device=device)
    k=1
    #Vertices with degree <=K
    B=C[(D[C]<=k)&(D[C]>0)]
    t1 = time.time()
    # tracer =VizTracer()
    # tracer.start()
    while C.shape[0]>1:
        # t3 = time.time()
        while B.shape[0]>0:
            D[B]=0
            K[B]=k
            m=multi_arange(II[B],CD[B])
            J=I[m]
            H=J[D[J]>0]
            H,cnt=torch.unique(H,return_counts=True)
            D[H]-=cnt
            B=H[D[H]<=k]
        # t4 = time.time()
        # Analyse_file.write("{},{}\n".format(k,t4-t3))
        k=k+1
        # indices of nodes that are relevant
        C=C[D[C]>=k]
        # indices of nodes that will be deleted
        B=C[D[C]==k]    
    t2 = time.time()
    # tracer.stop()
    # tracer.save()
    print('time: {}'.format(t2 - t1))
    print("-----max____k:",torch.max(K))
    # k,num=torch.unique(K,return_counts=True)
    # Analyse_file=open('/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/com-lj.ungraph_k.txt','w')
    # for i in range(len(k)):
    #     Analyse_file.write("{},{}\n".format(k[i],num[i]))
    # print("{}".format(K.device),K.tolist())
    
  
 
if __name__ == "__main__":
    main()
    
