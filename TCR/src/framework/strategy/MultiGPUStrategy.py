"""
A more complex strategy that runs computation in one process and run subgraph fetching + partitioning in another process.
"""
import sys
# sys.path.append('/home/lames/TCRGraph01/')
from . import Strategy
from ..partition import Partition
import torch
import torch.multiprocessing as mp
from src.type.Subgraph import Subgraph
import queue
from copy import deepcopy
import threading
import select
import time

class FetchPartitionsThread(threading.Thread):
    def __init__(self, lock, partition_queue, program, strategy, devices, **kwargs):
        super().__init__(**kwargs)
        self.lock = lock
        self.partition_queue = partition_queue
        self.program = program
        self.sub_in_degrees, self.sub_out_degrees = None, None
        self.partition = strategy.partition
        self.strategy = strategy
        self.devices = devices
    # run in a separate process; fetches partitions of graphs and put into queue
    def run(self):
        self.partition.set_num_partitions(self.partition.num_partitions * self.devices)
        partition_queue = self.partition_queue
        prog = self.program
        if self.sub_out_degrees is None and self.sub_in_degrees is None:
            sub_out_degrees, sub_in_degrees = None, None
            try:
                sub_out_degrees = prog.graph.out_degree(prog.graph.vertices)
            except:
                ...
            try:
                sub_in_degrees = prog.graph.in_degree(prog.graph.vertices)
            except:
                ...
            self.sub_out_degrees, self.sub_in_degrees = sub_out_degrees, sub_in_degrees
        else:
            sub_out_degrees, sub_in_degrees = self.sub_out_degrees, self.sub_in_degrees
        
        while True:
            with self.lock:
                while not self.strategy.changed_mask:
                    self.lock.wait()
                self.strategy.changed_mask = False
                activated_vertices = torch.nonzero(prog.activated_vertex_mask).squeeze()
            if activated_vertices is None or len(activated_vertices.shape) == 0 or activated_vertices.shape[0] == 0:
                partition_queue.put(None)
                break
            subgraph, indices = prog.graph.csr_subgraph(activated_vertices)
            self.partition.set_graph(subgraph)
            partitions = self.partition.generate_partitions()
            new_subgraphs = [[] for _ in range(self.devices)]
            for num, ((p, i), v) in enumerate(partitions):
                new_indices = indices[i]
                new_vertices = activated_vertices[v]
                sub_to_orig_vertices = torch.zeros_like(prog.graph.vertices, device=subgraph.device)
                sub_to_orig_vertices[new_vertices] = torch.arange(new_vertices.numel(), device=subgraph.device)
                sub_to_orig_edges = torch.zeros_like(prog.graph.edges, device=subgraph.device)
                sub_to_orig_edges[new_indices] = torch.arange(new_indices.numel(), device=subgraph.device)

                new_subgraph = Subgraph(p, new_vertices, new_indices, sub_out_degrees, sub_in_degrees, sub_to_orig_vertices, sub_to_orig_edges)
                new_subgraphs[num % self.devices].append(new_subgraph)
            partition_queue.put(new_subgraphs)
            del activated_vertices

# run with multithreading (same process with compute_on_gpu); moves partitions to GPU
class MoveToGPUThread(threading.Thread):
    def __init__(self, partition_queue, gpu_queues, devices, **kwargs):
        super().__init__(**kwargs)
        self.partition_queue = partition_queue
        self.gpu_queues = gpu_queues
        self.devices = devices
    def run(self):
        last_subgraphs = []
        while True:
            got_from_partition_queue = True
            subgraphs = None
            try:
                subgraphs = self.partition_queue.get_nowait()
            except queue.Empty:
                got_from_partition_queue = False

            if got_from_partition_queue:
                if subgraphs is None:
                    for q in self.gpu_queues:
                        q.put(None)
                    break
                last_subgraphs = subgraphs
            else:
                if len(last_subgraphs) == 0:
                    continue
            for device, subs in enumerate(last_subgraphs):
                for s in subs:
                    s.to(torch.device('cuda', device))
            for device, subs in enumerate(last_subgraphs):
                self.gpu_queues[device].join()
                for s in subs:
                    self.gpu_queues[device].put(s)
                self.gpu_queues[device].put(None)
        for q in self.gpu_queues:
            q.put(None)

# one process to compute subgraph on GPUs
class ComputeOnOneGPUProcess(mp.Process):
    def __init__(self, vertex_data, edge_data, activated_vertices_mask, changed_vertex_data_mask,
                 changed_edge_data_mask, gpu_queue, prog, message_pipe, rank, **kwargs):
        super().__init__(**kwargs)
        self.gpu_queue = gpu_queue
        self.prog = prog
        self.message_pipe = message_pipe    # to transfer data with parent process
        self.rank = rank
        self.activated_vertices_mask = activated_vertices_mask
        self.changed_vertex_data_mask = changed_vertex_data_mask
        self.changed_edge_data_mask = changed_edge_data_mask
        self.vertex_data = vertex_data
        self.edge_data = edge_data
        
    def run(self):
        # send message that I'm ready
        self.message_pipe.send(self.rank)
        last_none = False
        prog = self.prog
        activated_vertices_mask = self.activated_vertices_mask
        
        while True:
            # get a partitioned graph from queue
            part = self.gpu_queue.get() 
            self.gpu_queue.task_done()         
            if part is None:
                if last_none:
                    self.message_pipe.send(-self.rank - 1)
                    self.message_pipe.recv()
                    break
                # send message to parent process, and wait for message to continue to the next iteration
                self.message_pipe.send(self.rank)
                # update data
                self.message_pipe.recv()
                self.prog.vertex_data[:] = self.vertex_data.to(f'cuda:{self.rank}')
                self.prog.edge_data[:] = self.edge_data.to(f'cuda:{self.rank}')
                last_none = True
                continue
            last_none = False
            
            subgraph = part
            vertices = subgraph.sub_vertices
            indices = subgraph.sub_edges
            prog.set_graph(subgraph)

            g_nbrs, g_edges, g_ptr = prog.gather_nbrs(vertices)
            gd, gd_ptr = prog.gather(vertices, g_nbrs, g_edges, g_ptr)
            gsum = prog.sum(gd, gd_ptr)
            apply_d, apply_mask = prog.apply(vertices, gsum)
            prog.vertex_data[vertices] = torch.where(apply_mask,
                apply_d, prog.vertex_data[vertices])
            s_nbrs, s_edges, s_ptr = prog.scatter_nbrs(vertices)
            s_data, s_mask = prog.scatter(vertices, s_nbrs, s_edges, s_ptr, apply_d)
            if s_data is not None and s_mask is not None:
                prog.edge_data[indices] = torch.where(s_mask, s_data, prog.edge_data[indices])
                self.changed_edge_data_mask[indices.to('cuda')] = self.rank
            self.changed_vertex_data_mask[vertices.to('cuda')] = self.rank
                
            # update activated vertices
            # put new activated vertices into queue
            if not prog.not_change_activated:
                activated_vertices_mask = torch.logical_or(activated_vertices_mask, prog.next_activated_vertex_mask)
                
            # part.to('cpu')

        # prog.to('cpu')
        
class ComputeOnGPUThread(threading.Thread):
    def __init__(self, lock, gpu_queues, prog, strategy, devices, **kwargs):
        super().__init__(**kwargs)
        self.lock = lock
        self.gpu_queues = gpu_queues
        self.prog = prog
        self.vertex_data = None
        self.edge_data = None
        self.strategy = strategy
        self.devices = devices
        
        self.activated_vertices_mask = torch.zeros(prog.graph.vertices.numel(), dtype=torch.bool, device='cuda')
        
        self.changed_vertex_data_mask = torch.ones_like(self.prog.vertex_data, 
                                                              dtype=torch.int16, device=torch.device('cuda')) * -1
        self.changed_edge_data_mask = torch.ones_like(self.prog.edge_data, 
                                                            dtype=torch.int16, device=torch.device('cuda')) * -1
 
    def run(self):
        # start the processes according to the number of GPUs
        processes = []
        pipes = []
        progs = []
        for i in range(self.devices):
            conn1, conn2 = mp.Pipe()
            new_prog = deepcopy(self.prog)
            new_prog.to(torch.device('cuda', i))
            progs.append(new_prog)
            process = ComputeOnOneGPUProcess(self.prog.vertex_data, self.prog.edge_data, self.activated_vertices_mask, 
                                            self.changed_vertex_data_mask, self.changed_edge_data_mask, 
                                            self.gpu_queues[i], new_prog, conn1, i)
            processes.append(process)
            pipes.append(conn2)
            
        for p in processes:
            p.start()
        
        # wait for all processes are ready
        for p in pipes:
            p.recv()
        print('All processes ready.')
        
        t1 = time.time()
        
        ready_processes = set()
        exited_processes = set()
        while True:
            readable, _, _ = select.select(pipes, [], [])
            for r in readable:
                rank = r.recv()
                if rank >= 0:
                    ready_processes.add(rank)
                else:
                    exited_processes.add(rank)
                    pipes[-rank - 1].send(None)
            # are all processes ready?
            if len(ready_processes) == self.devices:
                ready_processes.clear()
                # if all ready, reduce vertex/edge data and update activated vertices
                for i in range(self.devices):
                    changed_vertex_data_mask = self.changed_vertex_data_mask == i
                    changed_edge_data_mask = self.changed_edge_data_mask == i
                    self.prog.vertex_data[changed_vertex_data_mask] = progs[i].vertex_data[changed_vertex_data_mask].to('cuda')
                    self.prog.edge_data[changed_edge_data_mask] = progs[i].edge_data[changed_edge_data_mask].to('cuda')
                self.changed_vertex_data_mask[:] = -1
                self.changed_edge_data_mask[:] = -1
                    
                if not self.prog.not_change_activated or (self.prog.nbr_update_freq != 0 and self.prog.curr_iter % self.prog.nbr_update_freq == 0): 
                    with self.lock:
                        self.lock.notify()
                        self.prog.activated_vertex_mask = self.activated_vertices_mask
                        self.strategy.changed_mask = True
                    self.prog.next_activated_vertex_mask = torch.zeros_like(self.prog.activated_vertex_mask, dtype=torch.bool)
                    self.activated_vertices_mask[:] = False
                    
                self.prog.curr_iter += 1
                self.prog.not_change_activated = False

                for i in range(self.devices):
                    # pipes[i].send((self.prog.vertex_data.to(f'cuda:{i}'), self.prog.edge_data.to(f'cuda:{i}')))
                    pipes[i].send(None)
            # exit if all processes have exited
            if len(exited_processes) == self.devices:
                break
        self.vertex_data = self.prog.vertex_data
        self.edge_data = self.prog.edge_data
        
        t2 = time.time()
        print(f"Time for calculation: {t2 - t1}s")
        
        for p in processes:
            p.join()
    
class MultiGPUStrategy(Strategy.Strategy):
    def __init__(self, partition: Partition, max_subgraphs_in_gpu=2, devices=2) -> None:
        super().__init__()
        self.partition = partition
        self.max_subgraphs_in_gpu = max_subgraphs_in_gpu
        self.changed_lock = threading.Condition() # lock: whether the activated_vertices has changed
        self.changed_mask = True
        self.devices = devices
            
    def compute(self):
        prog = self.program
        mp.set_start_method('spawn', force=True)
        # partitioned graph (still in CPU)
        partition_queue = queue.Queue()
        # partitioned graph transferred to GPU
        gpu_queues = [mp.JoinableQueue(self.max_subgraphs_in_gpu) for _ in range(self.devices)]
        # result queue: p3 -> main process
        p1 = FetchPartitionsThread(self.changed_lock, partition_queue, self.program, self, self.devices)
        p2 = MoveToGPUThread(partition_queue, gpu_queues, self.devices)
        p3 = ComputeOnGPUThread(self.changed_lock, gpu_queues, prog, self, self.devices)
        with self.changed_lock:
            self.changed_lock.notify()
        p1.start()
        p2.start()
        p3.start()
        
        p1.join()
        p2.join()
        p3.join()
        vertex_data, edge_data = p3.vertex_data, p3.edge_data
        
        return vertex_data, edge_data
        