"""
Run the Louvain community detection algorithm on the network. Assuming an undirected, weighted graph.
"""
import sys
sys.path.append('/home/lames/TCRGraph/')
from src.framework.GASProgram import GASProgram
from src.type.CSRGraph import CSRGraph
import torch
import time
import argparse
from torch_scatter import segment_csr
from src.framework.helper import batched_csr_selection

class LouvainPhaseOne(GASProgram):
    """
    This class conducts Phase 1 in the Louvain method. In Phase 1, the Louvain method greedily assigns each vertex to the community that maximizes the modularity gain. This is part of the Louvain method.
    """
    def __init__(self, graph: CSRGraph, vertex_data_type=torch.int32, edge_data_type=torch.int32, vertex_data=None, edge_data=None):
        # vertex_data: community id. by default, each vertex is assigned to its own community.
        if vertex_data is None:
            vertex_data = torch.arange(graph.num_vertices, dtype=vertex_data_type)
        # edge_data: edge weight. by default, each edge has weight 1.
        if edge_data is None:
            edge_data = torch.ones(graph.num_edges, dtype=edge_data_type)
        # we keep track of the total weight of each community.
        self.community_weights = torch.zeros(graph.num_vertices, dtype=edge_data_type)
        # we keep track of the vertices in each community.
        self.community_vertices = [[i] for i in range(graph.num_vertices)]
        # sum of all edge weights.
        self.sum_edge_weights = torch.sum(edge_data)
        self.changed = False
        self.overall_changed = False
        self.MODULARITY_UPDATE_THRESHOLD = 0.012
        super().__init__(graph, vertex_data_type, edge_data_type, vertex_data, edge_data, None)
    
    def to(self, device):
        self.community_weights = self.community_weights.to(device)
        super().to(device)
        
    def calculate_modularity_gain(self, vertex_u, vertex_v):
        """
        Calculate the modularity gain of assigning vertex_u to the community of vertex_v.
        See the Wikipedia entry for the modularity formula.
        
        :param int vertex_u: vertex to assign to a community
        :param int vertex_v: neighbor of vertex_u
        :return: modularity gain
        """
        new_community = self.vertex_data[vertex_v]
        if self.vertex_data[vertex_u] == new_community:
            return 0.0
        m = self.sum_edge_weights
        sum_in = self.community_weights[new_community]
        sum_tot = 0
        for vertex in self.community_vertices[new_community]:
            for e in self.graph.out_edges(torch.tensor([vertex]))[0][0]:
                sum_tot += self.edge_data[e]
                
        k_i = 0
        k_i_in = 0
        for v, e in zip(self.graph.out_nbrs(torch.tensor([vertex_u]))[0][0], self.graph.out_edges(torch.tensor([vertex_u]))[0][0]):
            k_i += self.edge_data[e]
            if self.vertex_data[v] == new_community:
                k_i_in += self.edge_data[e]
        result = ((sum_in + 2 * k_i_in) / (2 * m) - ((sum_tot + k_i) / (2 * m)) ** 2) \
            - (sum_in / (2 * m) - (sum_tot / (2 * m)) ** 2 - (k_i / (2 * m)) ** 2)
        return result

    def gather(self, vertices, nbrs, edges, ptr):
        modularity_gains = torch.zeros(len(nbrs), dtype=torch.float32, device=self.device)
        index = 0
        for i in range(len(vertices)):
            for j in range(ptr[i], ptr[i + 1]):
                mod_gain = self.calculate_modularity_gain(vertices[i], nbrs[j])
                modularity_gains[index] = mod_gain
                index += 1
        return torch.stack((modularity_gains, nbrs), dim=-1), ptr
    
    def sum(self, gathered_data, ptr):
        return segment_csr(gathered_data, ptr, reduce='max')
            
    def apply(self, vertices, gathered_sum):
        new_results = torch.zeros(len(vertices), dtype=self.vertex_data_type, device=self.device)
        index = 0
        for vertex_u, (mod_gain, to_vertex) in zip(vertices, gathered_sum):
            if mod_gain > self.MODULARITY_UPDATE_THRESHOLD:
                to_vertex = to_vertex.int()
                # if gain > 0, do the assignment vertex_u -> to_vertex.
                self.changed = True
                self.overall_changed = True
                original_community = self.vertex_data[vertex_u]
                new_community = self.vertex_data[to_vertex]
                self.community_vertices[original_community].remove(vertex_u)
                self.community_vertices[new_community].append(vertex_u)
                for nbr, e in zip(self.graph.out_nbrs(torch.tensor([vertex_u]))[0][0], self.graph.out_edges(torch.tensor([vertex_u]))[0][0]):
                    if nbr in self.community_vertices[original_community]:
                        self.community_weights[original_community] -= self.edge_data[e]
                    if nbr in self.community_vertices[new_community]:
                        self.community_weights[new_community] += self.edge_data[e]
                new_results[index] = new_community
            else:
                new_results[index] = self.vertex_data[vertex_u]
            index += 1
        return new_results, torch.ones(len(vertices), dtype=torch.bool, device=self.device)
            
    def scatter(self, vertices, nbrs, edges, ptr, apply_data):
        changed = apply_data != self.vertex_data[vertices]
        if self.nbr_update_freq > 0:
            if not torch.all(changed == 0):
                # get neighbors of changed
                starts, ends = ptr[changed], ptr[changed + 1]
                result, ptr = batched_csr_selection(starts, ends)
                all_neighbors = nbrs[result]
                self.activate(all_neighbors)
        else:
            if not torch.all(changed == 0):
                self.not_change_activated_next_iter()
        return None, None
    
    def gather_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr
    
    def scatter_nbrs(self, vertices):
        if self.nbr_update_freq == 0:
            out_nbrs, ptr = self.graph.all_out_nbrs_csr()
        else:
            out_nbrs, ptr = self.graph.out_nbrs_csr(vertices)
        return out_nbrs, None, ptr
        
def community_aggregation(graph: CSRGraph, vertex_data=None, edge_data=None):
    """
    This function aggregates the communities in the graph. (See Phase 2 of the Louvain algorithm)
    
    :param CSRGraph graph: graph to aggregate communities
    :param torch.Tensor vertex_data: vertex data (community assignment)
    :param torch.Tensor edge_data: edge data (edge weights)
    :return: new graph; new edge weights; a map from new graph vertices to according original graph vertices
    """
    new_edges_and_weights = {}
    new_vertices = {}
    for v in graph.vertices:
        for nbr, e in zip(graph.out_nbrs(torch.tensor([v]))[0][0], graph.out_edges(torch.tensor([v]))[0][0]):
            v = vertex_data[v].item()
            nbr = vertex_data[nbr].item()
            if v > nbr:
                v, nbr = nbr, v
            new_edges_and_weights[(v, nbr)] = new_edges_and_weights.get((v, nbr), 0) + edge_data[e]

    new_edge_starts = []
    new_edge_ends = []
    new_edge_weights = []
    
    for (v, nbr), weight in new_edges_and_weights.items():
        new_edge_starts.append(v)
        new_edge_ends.append(nbr)
        new_edge_weights.append(weight)
    
    new_graph, vtid = CSRGraph.edge_list_to_Graph(new_edge_starts, new_edge_ends, edge_attrs=[new_edge_weights])
    new_weights = new_graph.edge_attrs_tensor
    for v, comm in zip(graph.vertices, vertex_data):
        new_graph_vid = vtid[comm.item()]
        new_vertices[new_graph_vid] = new_vertices.get(new_graph_vid, []) + [v]
        
    return new_graph, new_weights, new_vertices

def louvain(graph: CSRGraph, edge_data=None, max_iter=100, cuda=False):
    """
    This function performs the Louvain algorithm on the graph.
    
    :param CSRGraph graph: graph to perform Louvain on
    :param torch.Tensor edge_data: edge data (edge weights)
    :param int max_iter: maximum number of iterations
    :return: vertex data (community assignment)
    """
    vertex_data = torch.arange(graph.num_vertices, dtype=torch.int32)
    if edge_data is None:
        edge_data = torch.ones(graph.num_edges, dtype=torch.int32)
    curr_graph = graph
    curr_edge_data = edge_data
    curr_graph_v_to_graph = {i.item(): [i.item()] for i in graph.vertices}
    for i in range(max_iter):
        # Phase 1: move each vertex to the community that gives the highest modularity gain.
        louvain = LouvainPhaseOne(curr_graph, edge_data=curr_edge_data)
        if cuda:
            curr_graph.to('cuda')
            louvain.to('cuda')
        louvain.compute()
        # If not changed, we do not proceed.
        if not louvain.overall_changed:
            break
        # write changes to curr_graph to graph.
        for v in curr_graph.vertices:
            v = v.item()
            for graph_v in curr_graph_v_to_graph[v]:
                vertex_data[graph_v] = louvain.vertex_data[v]
        # Phase 2: aggregate communities.
        curr_graph, curr_edge_data, new_vertices = community_aggregation(curr_graph, vertex_data=louvain.vertex_data, edge_data=curr_edge_data)
        curr_edge_data = curr_edge_data[0]
        # maintain curr_graph_v_to_graph
        new_curr_graph_v_to_graph = {}
        for v in new_vertices:
            new_curr_graph_v_to_graph[v] = []
            for graph_v in new_vertices[v]:
                graph_v = graph_v.item()
                new_curr_graph_v_to_graph[v] += curr_graph_v_to_graph[graph_v]
        curr_graph_v_to_graph = new_curr_graph_v_to_graph
        
    return vertex_data

def main():
    # Assume that your graph is undirected
    # Assume that one weight exists for each edge
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, help='path to graph', required=True)
    parser.add_argument('--output', type=str, help='output path to vertex results', required=True)
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--maxsteps', type=int, help='maximum number of iterations', default=50)
    args = parser.parse_args()
    
    print('reading graph...', end=' ', flush=True)
    graph, _ = CSRGraph.read_graph(args.graph, split=None)
    data = graph.edge_attrs_tensor
    if len(data) != 1:
        print('No weights provided or too many weights provided. Default to w(e)=1')
        data = torch.ones((graph.num_edges,), dtype=torch.int32)
    else:
        data = data[0]
    print('Done!')
    
    t1 = time.time()
    vertex_data = louvain(graph, edge_data=data, cuda=args.cuda, max_iter=args.maxsteps)
    t2 = time.time()
    
    print('Completed! {}s time elapsed. Outputting results...'.format(t2 - t1))
    # output results
    with open(args.output, 'w') as f:
        for i in range(len(vertex_data[:])):
            f.write(str(vertex_data[i].item()) + '\n')

if __name__ == '__main__':
    main()
