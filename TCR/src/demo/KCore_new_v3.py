import torch
import sys
sys.path.append('/root/autodl-tmp//TCRGraph-tcr_gas')
from src.framework.GASProgram import GASProgram
from src.type.CSRCGraph import CSRCGraph, CSRGraph
import argparse
from torch_scatter import segment_csr, scatter_add, segment_coo
import time
# from viztracer import VizTracer

import logging 
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO)
DEBUG = False


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
    filename ='soc-twitter'
    path='/root/autodl-tmp/dataset/{}'.format(filename)
    graph= CSRGraph.read_graph_bin(path,is_long=False)
    logging.info('graph vertex {} edges {}'.format(graph.num_vertices, graph.num_edges))
    graph.to(device)
    # print(graph.out_degrees)
    print('read graph done')

    #--------
    # Analyse_file=open('/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/large_twitch_edges.csv_ana.txt','a')

    I=graph.columns.to(torch.long).to(device)
    D=graph.out_degrees.to(torch.long).to(device)
    II=graph.row_ptr.to(torch.long)
    K=torch.zeros(graph.num_vertices, dtype=torch.long, device=device)
    # copy of D to remain unchanged
    CD=D.clone()
    C=torch.arange(graph.num_vertices,dtype=torch.long, device=device)
    k=1
    #Vertices with degree <=K
    B=C[D[C]<=k]
    t1 = time.time()
    # tracer =VizTracer()
    # tracer.start()
    while C.shape[0]>1:
        # t3 = time.time()
        while B.shape[0]>0:
            D[B]=0
            K[B]=k
            m=multi_arange(II[B],CD[B])
            # m,_=find_nbr(II[B],II[B+1]) ,multi_arange 更快
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
    # Analyse_file=open('/root/autodl-tmp/tcr/TCRGraph-tcr_gas/Data/Analyse_file/com-lj.ungraph_k.txt','w')
    # for i in range(len(k)):
    #     Analyse_file.write("{},{}\n".format(k[i],num[i]))
    print("maxk:{}".format(torch.max(K)))
    
  
 
if __name__ == "__main__":
    main()
    
