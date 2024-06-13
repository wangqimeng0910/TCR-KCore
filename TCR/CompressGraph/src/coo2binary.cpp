#include <bits/types/FILE.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_map>

struct pair_hash
{
    template<class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

void directed2binary(std::string inpath, std::string outpath)
{
    std::vector<std::pair<int,int>> edges;
    std::vector<int> vertices;
    std::unordered_map<int, int> hash_is_exit;
    std::ifstream infile(inpath);
    int src, dst;
    while(infile >> src >> dst)
    {
        if(!hash_is_exit[src]) 
        {
            hash_is_exit[src] = 1;
            vertices.emplace_back(src);
        }
        if(!hash_is_exit[dst]) 
        {
            hash_is_exit[dst] = 1;
            vertices.emplace_back(dst);
        }
        edges.emplace_back(std::make_pair(src, dst));
    }
    infile.close();
    std::cout << "read data done" << std::endl;
    // renumber
    sort(vertices.begin(), vertices.end());
    int vertex_cnt = vertices.size();
    hash_is_exit.clear();
    std::unordered_map<int, int> hash_renumber;
    for(int i = 0; i < vertices.size(); i++)
    {
        hash_renumber[vertices[i]] = i;
    }
    for(auto& pair : edges)
    {
        pair.first = hash_renumber[pair.first];
        pair.second = hash_renumber[pair.second];
    }
    vertices.clear();
    std::cout << "renumber done" << std::endl;

    // generate csr and csc format
    std::vector<int> vlist(vertex_cnt + 1, 0);
    std::vector<int> elist;

    // first generate csr format
    sort(edges.begin(), edges.end());
    for(auto pair : edges)
    {
        vlist[pair.first + 1] ++;
        elist.emplace_back(pair.second);
    }
    for(int i = 0; i < vertex_cnt; i++)
    {
        vlist[i + 1] += vlist[i];
    }
    std::cout << "generate csr format done" << std::endl;

    // write csr format to binary file
    std::string csr_vlist_path = outpath + "/csr/vlist.bin";
    std::string csr_elist_path = outpath + "/csr/elist.bin";
    FILE* csr_vlist_file = fopen(csr_vlist_path.c_str(), "w");
    FILE* csr_elist_file = fopen(csr_elist_path.c_str(), "w");
    std::cout << "write csr format to binary file start" << std::endl;
    fwrite(&vlist[0], sizeof(int), vlist.size(), csr_vlist_file);
    fwrite(&elist[0], sizeof(int), elist.size(), csr_elist_file);
    std::cout << "write csr format to binary file done" << std::endl;

    // second generate csc format
    sort(edges.begin(), edges.end(), [](std::pair<int,int> a, std::pair<int,int> b){return a.second < b.second;});
    std::fill(vlist.begin(), vlist.end(), 0);
    elist.clear();
    for(auto pair : edges)
    {
        vlist[pair.second + 1] ++;
        elist.emplace_back(pair.first);
    }
    for(int i = 0; i < vertex_cnt; i++)
    {
        vlist[i + 1] += vlist[i];
    }
    std::cout << "generate csc format done" << std::endl;

    // write csc format to binary file
    std::string csc_vlist_path = outpath + "/csc/vlist.bin";
    std::string csc_elist_path = outpath + "/csc/elist.bin";
    FILE* csc_vlist_file = fopen(csc_vlist_path.c_str(), "w");
    FILE* csc_elist_file = fopen(csc_elist_path.c_str(), "w");
    fwrite(&vlist[0], sizeof(int), vlist.size(), csc_vlist_file);
    fwrite(&elist[0], sizeof(int), elist.size(), csc_elist_file);
    std::cout << "write csc format to binary file done" << std::endl;

    return;
}

void undirected2binary(std::string inpath, std::string outpath)
{
    std::vector<std::pair<int,int>> edges;
    std::vector<int> vertices;
    std::unordered_map<int, int> vertex_is_exit;
    std::unordered_map<std::pair<int,int>, int, pair_hash> edge_is_exit;
    std::ifstream infile(inpath);
    int src, dst;
    while(infile >> src >> dst)
    {
        if(!vertex_is_exit[src]) 
        {
            vertex_is_exit[src] = 1;
            vertices.emplace_back(src);
        }
        if(!vertex_is_exit[dst]) 
        {
            vertex_is_exit[dst] = 1;
            vertices.emplace_back(dst);
        }
        if(!edge_is_exit[std::make_pair(src, dst)])
        {
            edge_is_exit[std::make_pair(src, dst)] = 1;
            edges.emplace_back(std::make_pair(src, dst));
        }
        if(!edge_is_exit[std::make_pair(dst, src)])
        {
            edge_is_exit[std::make_pair(dst, src)] = 1;
            edges.emplace_back(std::make_pair(dst, src));
        }
    }
    infile.close();

    // renumber
    sort(vertices.begin(), vertices.end());
    int vertex_cnt = vertices.size();
    vertex_is_exit.clear();
    std::unordered_map<int, int> hash_renumber;
    for(int i = 0; i < vertices.size(); i++)
    {
        hash_renumber[vertices[i]] = i;
    }
    for(auto& pair : edges)
    {
        pair.first = hash_renumber[pair.first];
        pair.second = hash_renumber[pair.second];
    }
    vertices.clear();

    // generate csr and csc format
    std::vector<int> vlist(vertex_cnt + 1, 0);
    std::vector<int> elist;
    sort(edges.begin(), edges.end());
    for(auto pair : edges)
    {
        vlist[pair.first + 1] ++;
        elist.emplace_back(pair.second);
    }
    for(int i = 0; i < vertex_cnt; i++)
    {
        vlist[i + 1] += vlist[i];
    }
    // write csr format to binary file
    std::string csr_vlist_path = outpath + "/csr/vlist.bin";
    std::string csr_elist_path = outpath + "/csr/elist.bin";
    std::string csc_vlist_path = outpath + "/csc/vlist.bin";
    std::string csc_elist_path = outpath + "/csc/elist.bin";
    FILE* csr_vlist_file = fopen(csr_vlist_path.c_str(), "w");
    FILE* csr_elist_file = fopen(csr_elist_path.c_str(), "w");
    FILE* csc_vlist_file = fopen(csc_vlist_path.c_str(), "w");
    FILE* csc_elist_file = fopen(csc_elist_path.c_str(), "w");
    fwrite(&vlist[0], sizeof(int), vlist.size(), csr_vlist_file);
    fwrite(&elist[0], sizeof(int), elist.size(), csr_elist_file);
    fwrite(&vlist[0], sizeof(int), vlist.size(), csc_vlist_file);
    fwrite(&elist[0], sizeof(int), elist.size(), csc_elist_file);
    return ;
}   

int main(int argc, char** argv)
{
    std::string inpath = argv[1];
    std::string outpath = argv[2];
    // 0 is undirected, 1 is directed
    int is_directed = atoi(argv[3]);

    if(is_directed)
    {
        std::cout << "reading directed graph data to binary " << std::endl;
        directed2binary(inpath, outpath);
    }
    else
    {
        std::cout << "reading undirected graph data to binary " << std::endl;
        undirected2binary(inpath, outpath);
    }
    
    return 0;
}