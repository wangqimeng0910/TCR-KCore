import torch
from torch_scatter import segment_csr, segment_coo
import random
import sys
sys.path.append('/root/Project/TCRGraph/')
import time
from src.type.CSRCGraph import CSRCGraph
if __name__ == "__main__":
    # vertex_cnt = 3774768
    # edge_cnt = 16518948

    # columns = torch.rand(edge_cnt, dtype=torch.float16).cuda()
    # index = torch.ones(vertex_cnt + 1, dtype=torch.long).cuda()

    # list = []
    # list.append(0)
    # for i in range(vertex_cnt - 1):
    #     a = random.randint(0, edge_cnt)
    #     edge_cnt -= a
    #     list.append(a)

    # list.append(edge_cnt)

    # for i in range(vertex_cnt):
    #     list[i + 1] += list[i]
    
    # index = torch.tensor(list, dtype=torch.long).cuda()

    # import time
    # t1 = time.time()
    # for i in range(20):
    #     segment_csr(columns, index, reduce='sum')
    # # torch.cuda.synchronize()
    # t2 = time.time()

    # print(t2 - t1)

    
    # path = '/root/Project/TCRGraph/data/wiki-Vote.txt'
    # path = '/root/Project/TCRGraph/data/graph500-22/csrc_data'
    # path = '/root/autodl-tmp/data/LiveJournal/csrc_data'
    # path = '/root/Project/TCRGraph/data/datagen-7_5-fb/data'
    # path = '/root/autodl-tmp/data/web-BerkStan/csrc_data'
    # path = '/root/autodl-tmp/data/wiki-walk/wiki-Talk.txt'
    path = '/root/autodl-tmp/data/cnr-2000/csrc_data'
    path = '/root/autodl-tmp/data/uk-2007-05/csrc_data'
    path = '/root/autodl-tmp/data/in-2004/csrc_data'
    path = '/root/autodl-tmp/data/eu-2005/csrc_data'
    path = '/root/autodl-tmp/data/hollywood-2009/csrc_data'
    # path = '/root/autodl-tmp/data/cit-Patents/csrc_data'
    path = '/root/autodl-tmp/data/eu-2015-host/csrc_data'
    t1 = time.time()
    graph = CSRCGraph.read_csrc_graph_bin(path)
    # graph, _, _ = CSRCGraph.read_graph(path)
    t2 = time.time()
    print('read graph time', t2 - t1)
    graph.pin_memory()
    graph.to('cuda')
    print('num_vertex', graph.num_vertices)
    deads_nodes = torch.where(graph.csr.out_degrees == 0)[0]
    print(deads_nodes)
    row_ptr = graph.csc.csr.row_ptr
    columns = graph.csc.csr.columns
    print(row_ptr.shape, " ", columns.shape)
    # 出度为0的节点 会导致inf 但是这些点也不会出现在columns中
    t1 = time.time()
    frac = 1 / graph.csr.out_degrees * 0.85
    frac[deads_nodes] = 0.85 / graph.num_vertices
    torch.cuda.synchronize()
    t2 = time.time()
    print('pre time', t2 - t1)
    frac.to(torch.float32)
    # print(frac)
    xs = 0.85 / graph.num_vertices
    # print(xs)
    # frac = frac[columns].to(torch.float32)
    # print(frac.dtype)
    ans = torch.ones(graph.num_vertices, dtype=torch.float32).cuda()
    print(torch.sum(ans))
    index = torch.arange(graph.num_vertices, dtype=torch.long).cuda().\
           repeat_interleave(torch.diff(row_ptr))
    index = torch.cat((index, torch.tensor([graph.num_vertices - 1], dtype=torch.long).cuda()))
    columns = torch.cat((columns, torch.tensor([0], dtype=torch.long).cuda()))
    # t1 = time.time()
    # for i in range(25):
    #     ans = segment_csr(values, row_ptr, reduce='sum')
    #     ans = ans * 0.85 + 0.15 
    #     values = ans[columns]
    # torch.cuda.synchronize()
    # t2 = time.time()

    # t1 = time.time()
    # for i in range(10):
    #     ans = ans * frac
    #     data = ans[columns]
    #     x = torch.sum(ans[deads_nodes]) * xs + 0.15        
    #     ans = segment_coo(data, index, reduce='sum')
    #     ans = ans + x
    # torch.cuda.synchronize()
    # t2 = time.time()
    # print('all time', t2 - t1, "  ", torch.sum(ans).item())
    from viztracer import VizTracer
    
    # data = torch.rand_like(columns, dtype=torch.float32).cuda()
    tracer = VizTracer()
    tracer.start()
    t1 = time.time()
    for i in range(150):
        x = torch.sum(ans[deads_nodes], dim=0) * xs
        ans = ans * frac
        data = ans[columns]
        # data = torch.index_select(ans, 0, columns)
        # ans = segment_csr(data, row_ptr, reduce='sum')
        ans = segment_coo(data, index, reduce='sum', dim_size=graph.num_vertices)
        ans = ans + 0.15 + x
    torch.cuda.synchronize()
    t2 = time.time()
    tracer.stop()
    tracer.save()
    print('all time', t2 - t1, "  ", torch.sum(ans).item())





