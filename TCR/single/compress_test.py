"""
Test for compress graph
"""
import torch
from torch_scatter import segment_csr, segment_coo, scatter_add
import random
import sys
sys.path.append('/root/Project/TCRGraph/')
import time
import logging  
from src.type.CSRCGraph import CSRCGraph
from src.type.CSRGraph import CSRGraph
from src.type.CompressGraph import CompressGraph
if __name__ == "__main__":
    # path = '/root/Project/TCRGraph/data/wiki-Vote.txt'
    path = '/root/Project/TCRGraph/data/graph500-22/csrc_data'
    # path = '/root/autodl-tmp/data/LiveJournal/csrc_data'
    # path = '/root/Project/TCRGraph/data/datagen-7_5-fb/data'
    # path = '/root/autodl-tmp/data/web-BerkStan/csrc_data'
    # path = '/root/Project/TCRGraph/data/cit-Patents/csrc_data'
    # path = '/root/autodl-tmp/data/wiki-walk/wiki-Talk.txt'

    path = '/root/autodl-tmp/data/cnr-2000/csrc_data'
    path = '/root/autodl-tmp/data/uk-2007-05/csrc_data'
    path = '/root/autodl-tmp/data/in-2004/csrc_data'
    # path = '/root/autodl-tmp/data/eu-2005/csrc_data'
    # path = '/root/autodl-tmp/data/hollywood-2009/csrc_data'
    path = '/root/autodl-tmp/data/eu-2015-host/csrc_data'
    graph = CSRCGraph.read_csrc_graph_bin(path)
    print('read csrc graph done')
    graph.to('cuda')
    deads_nodes = torch.where(graph.csr.out_degrees == 0)[0]
    path = '/root/autodl-tmp/data/uk-2007-05/compress_data'
    path = '/root/autodl-tmp/data/in-2004/compress_data'
    # path = '/root/autodl-tmp/data/eu-2005/compress_data'
    # path = '/root/autodl-tmp/data/hollywood-2009/compress_data'
    path = '/root/autodl-tmp/data/eu-2015-host/compress_data'
    cgraph = CompressGraph.read_compress_graph_bin(path)
    print('read cgraph done')
    cgraph.to('cuda')
    print(cgraph.depth)
    print(cgraph.length_every_depth)
    print(cgraph.num_vertex)
    print(cgraph.num_rule)

    # vertex_data 325557 the output result
    # vertex_data = torch.ones(cgraph.num_vertex, dtype=torch.float32).cuda()

    # vertex_and_rule_data
    vertex_and_rule_data = torch.ones(cgraph.num_vertex + cgraph.num_rule, dtype=torch.float32).cuda()
    # stroe the output pr value for vertex and rule
    vertex_and_rule_pr = torch.zeros(cgraph.num_vertex + cgraph.num_rule, dtype=torch.float32).cuda()
    # lvr = cgraph.num_vertex + cgraph.num_rule(length of vertex and rule)
    lvr = cgraph.num_vertex + cgraph.num_rule

    graph.csr.out_degrees.to('cuda')
    deads_nodes.to('cuda')
    degress = 1 / graph.csr.out_degrees * 0.85
    xs = 0.85 / graph.num_vertices
    degress[deads_nodes] = 0.85 / graph.num_vertices
    sum = torch.sum(degress)
    print(degress)
    print("degrees sum: {}".format(sum))
    from viztracer import VizTracer
    tracer = VizTracer()
    tracer.start()

    print(cgraph.length_every_depth)
    print(cgraph.depth)
    # do one iteration
    t1 = time.time()
    for i in range(150):
        x = None
        for j in range(cgraph.depth + 1):
            # print(i)
            if j == 0:
                x = torch.sum(vertex_and_rule_data[deads_nodes]) * xs + 0.15
                # vertex_and_rule_pr.zero_()
                vertex_and_rule_pr[:cgraph.num_vertex] = vertex_and_rule_data[:cgraph.num_vertex] * degress 
                vertex_and_rule_data.zero_()
            # print('depth {}  length {}'.format(i, cgraph.length_every_depth[i + 1] - cgraph.length_every_depth[i]))
            coo_v0 = cgraph.coo0[cgraph.length_every_depth[j]:cgraph.length_every_depth[j + 1]]
            coo_v1 = cgraph.coo1[cgraph.length_every_depth[j]:cgraph.length_every_depth[j + 1]]
            values = vertex_and_rule_pr[coo_v1]
            if j == 0 and torch.any(coo_v1 >= cgraph.num_vertex):
                print("error")
            segment_coo(values, coo_v0, reduce='sum', dim_size=lvr, out=vertex_and_rule_data)
            # vertex_and_rule_data += tmp
            vertex_and_rule_pr[cgraph.num_vertex:] = vertex_and_rule_data[cgraph.num_vertex:]
            # tmp = scatter_add(values, coo_v1, dim_size=lvr)
            # vertex_and_rule_data += tmp
        
        vertex_and_rule_data[:cgraph.num_vertex] += x
        # print("iteration: {} sum: {}".format(i, torch.sum(vertex_and_rule_data[:cgraph.num_vertex])))
        # x = torch.sum(vertex_and_rule_data[deads_nodes]) * xs + 0.15
    t2 = time.time()
    torch.cuda.synchronize()
    tracer.stop()
    tracer.save('result.json')
    print('time', t2 - t1, torch.sum(vertex_and_rule_data[:cgraph.num_vertex]))


