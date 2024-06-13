#include "../core/compression/basics.h"
#include "../core/compression/hash.h"
#include "../core/compression/heap.h"
#include "../core/compression/records.h"
#include "../core/io.hpp"
#include "../core/util.hpp"
#include <algorithm>
#include <ostream>
#include <string>
#include <fstream>

int main(int argc, char** argv)
{
    std::string pvlist = argv[1];
    std::string pelist = argv[2];
    std::string outpath = argv[3];
    std::vector<int> csr_vlist, csr_elist;
    int v_cnt = read_binary2vector(pvlist, csr_vlist);
    int e_cnt = read_binary2vector(pelist, csr_elist);
    std::cout << "v_cnt: " << v_cnt << " e_cnt: " << e_cnt << std::endl;
    std::cout << csr_vlist[csr_vlist.size() - 1] << std::endl;

    // write data in coo formate
    std::ofstream outfile(outpath);
    std::cout << "csr_vlist.size(): " << csr_vlist.size() << std::endl;
    std::cout << "csr_elist.size(): " << csr_elist.size() << std::endl;
    for(int i = 0; i < csr_vlist.size() - 1; i++)
    {
        for(int j = csr_vlist[i]; j < csr_vlist[i + 1]; j++)
        {
            outfile << i << " " << csr_elist[j] << std::endl;
        }
    }
    
    // DEBUG 
    std::cout << csr_vlist[325556] << " " << csr_vlist[325557] << " " << csr_vlist[325558] << " " << csr_vlist[325559] << std::endl;
    
    // csr_vlist.pop_back();
    // csr_vlist.pop_back();

    // // write to origin data in binary formate
    // FILE *fvlist;
    // FILE *felist;
    // fvlist = fopen("/root/Project/CompressGraph/dataset/cnr-2000/origin/csr_vlist.bin", "w");
    // felist = fopen("/root/Project/CompressGraph/dataset/cnr-2000/origin/csr_elist.bin", "w");

    // // write data to binary file
    // fwrite(&csr_vlist[0], sizeof(int), csr_vlist.size(), fvlist);
    // fwrite(&csr_elist[0], sizeof(int), csr_elist.size(), felist);

    // change from csr to csc
    std::vector<int> csc_vlist(v_cnt, 0);
    std::vector<int> csc_elist;

    std::vector<std::pair<int, int>> edges;
    for(int i = 0; i < csr_vlist.size() - 1; i++)
    {
        for(int j = csr_vlist[i]; j < csr_vlist[i + 1]; j++)
        {
            edges.push_back(std::make_pair(csr_elist[j], i));
        }
    }
    std::sort(edges.begin(), edges.end());
    for(int i = 0; i < edges.size(); i++)
    {
        csc_vlist[edges[i].first + 1] ++;
        csc_elist.emplace_back(edges[i].second);
    }
    for(int i = 0; i < v_cnt - 1; i++)
    {
        csc_vlist[i + 1] += csc_vlist[i];
    }

    FILE *fvlist;
    FILE *felist;
    fvlist = fopen("/root/Project/CompressGraph/dataset/cnr-2000/origin/csr_vlist.bin", "w");
    felist = fopen("/root/Project/CompressGraph/dataset/cnr-2000/origin/csr_elist.bin", "w");

    // write data to binary file
    fwrite(&csc_vlist[0], sizeof(int), csr_vlist.size(), fvlist);
    fwrite(&csc_elist[0], sizeof(int), csr_elist.size(), felist);
}