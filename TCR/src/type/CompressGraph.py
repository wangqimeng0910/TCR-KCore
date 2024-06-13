"""
A Graph type implemented with Compressed Graph
"""
import torch
class CompressGraph():
    def __init__(self, coo0=None, coo1=None, depth=None, num_vertex=None,  length_every_depth=None, num_edge = None, num_rule = 0, directed=True) -> None:
        self.depth = depth
        self.directed = directed
        self.num_vertex = num_vertex
        self.num_edge = num_edge
        self.num_rule = num_rule
        self.coo0 = coo0
        self.coo1 = coo1
        self.length_every_depth = length_every_depth
    
    def to(self, *args, **kwargs):
        if self.coo0 is not None:
            self.coo0 = self.coo0.to(*args, **kwargs)

        if self.coo1 is not None:
            self.coo1 = self.coo1.to(*args, **kwargs)
        
        if self.length_every_depth is not None:
            self.length_every_depth = self.length_every_depth.to(*args, **kwargs)
    
    @staticmethod
    def read_compress_graph(f, split=None):
        """
        读取compress graph 文件
        """
        print('-------- {} ------------'.format(f))
        f = open(f, 'r')
        lines = f.readlines()
        print('lines {}'.format(len(lines)))
        info = lines[0].split()
        vertex_cnt = int(info[0])
        rule_cnt = int(info[1])
        depth = int(info[2])
        print('vertex_cnt: {} rule_cnt: {} depth: {}'.format(vertex_cnt, rule_cnt, depth))
        length_every_depth = [0]
        coo0 = []
        coo1 = []
        for i in range(depth + 1):
            v0 = lines[2 * i + 1].split(' ')
            v1 = lines[2 * i + 2].split(' ')
            v0 = [int(v) for v in v0[:-1]]
            v1 = [int(v) for v in v1[:-1]]
            length_every_depth.append(length_every_depth[-1] + len(v0))
            coo0.extend(v0)
            coo1.extend(v1)

        return CompressGraph(
            depth=depth, coo0=torch.tensor(coo0, dtype=torch.int64), coo1=torch.tensor(coo1, dtype=torch.int64),
            num_vertex=vertex_cnt, length_every_depth=torch.tensor(length_every_depth, dtype=torch.int64),
            num_rule=rule_cnt, directed=True
        )        
        
    @staticmethod
    def read_compress_graph_bin(f):
        """
        read compress graph from binary file
        """
        print('-------- {} ------------'.format(f))
        import os
        import numpy as np

        file = open(f, 'rb')
        size = os.path.getsize(f)
        size = size // 4

        vertex_cnt = np.fromfile(file, dtype=np.int32, count=1)[0]
        rule_cnt = np.fromfile(file, dtype=np.int32, count=1)[0]
        depth = np.fromfile(file, dtype=np.int32, count=1)[0]
        length_every_depth = [0]
        coo0 = np.array([], dtype=np.int32)
        coo1 = np.array([], dtype=np.int32)
        print('vertex_cnt: {} rule_cnt: {} depth: {}'.format(vertex_cnt, rule_cnt, depth))
        for i in range(depth + 1):
            length = np.fromfile(file, dtype=np.int32, count=1)[0]
            length_every_depth.append(length_every_depth[-1] + length)
            # read coo0 and coo1
            c0 = np.fromfile(file, dtype=np.int32, count=length)
            c1 = np.fromfile(file, dtype=np.int32, count=length)

            coo0 = np.concatenate((coo0, c0))
            coo1 = np.concatenate((coo1, c1))
            print('{} done'.format(i))

        return CompressGraph(
            depth=depth, coo0=torch.tensor(coo0, dtype=torch.int64), coo1=torch.tensor(coo1, dtype=torch.int64),
            num_vertex=vertex_cnt, length_every_depth=torch.tensor(length_every_depth, dtype=torch.int64),
            num_rule=rule_cnt, directed=True
        )