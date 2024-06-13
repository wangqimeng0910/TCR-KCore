#include <bits/types/clock_t.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <time.h>

typedef std::pair<uint, uint> PII; 
// #define DEBUG = 1

void read_directed_data(std::string inpath, std::string outpath)
{
    clock_t start, end;
    std::cout << "read data path: " << inpath << std::endl;
    std::cout << "write data path: " << outpath << std::endl;
    start = clock();
    // renumber the vertex
    std::ifstream infile(inpath);
    
    std::unordered_map<uint, uint> hash_is_exit;
    std::vector<uint> vertices;
    std::vector<PII> edges;
    uint src, dst;

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
    hash_is_exit.clear();
    std::cout << "read data done" << std::endl;

    // sort the vertex to renumber
    sort(vertices.begin(), vertices.end());
    uint vertex_cnt  = vertices.size();
    uint edge_cnt = edges.size();

    #ifdef DEBUG 
        std::cout << "vertex_cnt " << vertex_cnt << std::endl;
    #endif

    std::unordered_map<uint, uint> hash_renumber;
    for(uint i = 0; i < vertices.size(); i++)
    {
        hash_renumber[vertices[i]] = i;
    }

    // renumber the vertex
    for(auto& pair : edges)
    {
        pair.first = hash_renumber[pair.first];
        pair.second = hash_renumber[pair.second];
    }
    std::cout << "renumber done" << std::endl;

    #ifdef DEBUG
        for(const auto& pair : hash_renumber)
        {
            std::cout << pair.first << " " << pair.second << std::endl;
        }
        for(auto pair : edges)
        {
            std::cout << pair.first << " " << pair.second << std::endl;
        }
    #endif
    sort(edges.begin(), edges.end());

    // generate the new csr graph
    std::vector<uint> csr_vlist(vertex_cnt + 1, 0);
    std::vector<uint> csr_elist;
    for(auto pair : edges)
    {
        csr_vlist[pair.first + 1] ++;
        csr_elist.emplace_back(pair.second);
    }
    for(uint i = 0; i < vertex_cnt; i++)
    {
        csr_vlist[i + 1] += csr_vlist[i];
    }
    // generate the new csc graph
    sort(edges.begin(), edges.end(), [](PII a, PII b){
        return a.second == b.second ? a.first < b.first : a.second < b.second;});
    
    std::vector<uint> csc_vlist(vertex_cnt + 1, 0);
    std::vector<uint> csc_elist;
    for(auto pair : edges)
    {
        csc_vlist[pair.second + 1] ++;
        csc_elist.emplace_back(pair.first);
    }
    for(uint i = 0; i < vertex_cnt ; i++)
    {
        csc_vlist[i + 1] += csc_vlist[i];
    }
    std::cout << "csc done" << std::endl;
    // debug
    #ifdef DEBUG
        for(uint v : csr_vlist) std::cout << v << " ";
        std::cout << std::endl;
        for(uint e : csr_elist) std::cout << e << " ";
        std::cout << std::endl;

        for(uint v : csc_vlist) std::cout << v << " ";
        std::cout << std::endl;
        for(uint e : csc_elist) std::cout << e << " ";
        std::cout << std::endl;
    #endif

    // write the csr and csc graph
    std::cout << "start write data into four binary file" << std::endl;

    // std::ofstream outfile(outpath);
    // std::cout << vertex_cnt << " " << edge_cnt << std::endl;
    // outfile << vertex_cnt << " " << edge_cnt << std::endl;
    // for(uint v : csr_vlist) outfile << v << " ";
    // outfile << std::endl;
    // for(uint e : csr_elist) outfile << e << " ";
    // outfile << std::endl;
    // for(uint v : csc_vlist) outfile << v << " ";
    // outfile << std::endl;
    // for(uint e : csc_elist) outfile << e << " ";
    // outfile << std::endl;

    // convinent for pandas read
    // for(int i = 0; i < csr_elist.size(); i ++)
    // {
    //     if(i == csr_elist.size() - 1)
    //         outfile << csr_elist[i] << std::endl;
    //     else
    //         outfile << csr_elist[i] << " ";
    // }
    // for(int i = 0; i < csr_vlist.size(); i ++)
    // {
    //     if(i == csr_vlist.size() - 1)
    //         outfile << csr_vlist[i] << std::endl;
    //     else
    //         outfile << csr_vlist[i] << " ";
    // }
    // for(int i = 0; i < csc_elist.size(); i ++)
    // {
    //     if(i == csc_elist.size() - 1)
    //         outfile << csc_elist[i] << std::endl;
    //     else
    //         outfile << csc_elist[i] << " ";
    // }
    // for(int i = 0; i < csc_vlist.size(); i ++)
    // {
    //     if(i == csc_vlist.size() - 1)
    //         outfile << csc_vlist[i] << std::endl;
    //     else
    //         outfile << csc_vlist[i] << " ";
    // }

    FILE* csrv = fopen((outpath + "/csr_vlist.bin").c_str(), "w");
    FILE* csre = fopen((outpath + "/csr_elist.bin").c_str(), "w");
    FILE* cscv = fopen((outpath + "/csc_vlist.bin").c_str(), "w");
    FILE* csce = fopen((outpath + "/csc_elist.bin").c_str(), "w");
    fwrite(&csr_vlist[0], sizeof(int), csr_vlist.size(), csrv);
    fwrite(&csr_elist[0], sizeof(int), csr_elist.size(), csre);
    fwrite(&csc_vlist[0], sizeof(int), csc_vlist.size(), cscv);
    fwrite(&csc_elist[0], sizeof(int), csc_elist.size(), csce);
    fclose(csrv);
    fclose(csre);
    fclose(cscv);
    fclose(csce);
    std::cout << "write data done" << std::endl;
    end = clock();
    std::cout << "time: " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;
}

int main(int argc, char** argv)
{
    std::string inpath = argv[1];
    std::string outpath = argv[2];
    read_directed_data(inpath, outpath);
}