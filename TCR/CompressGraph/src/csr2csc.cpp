/*
transform csr format to csc format
the original csc format data can't be compress well(usually compress ratio < 1)
so convert the good compress csr data to csc data
*/
#include <bits/types/FILE.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "../core/compression/basics.h"
#include "../core/compression/hash.h"
#include "../core/compression/heap.h"
#include "../core/compression/records.h"
#include "../core/io.hpp"
#include "../core/util.hpp"


int main(int argc, char** argv)
{
    /*
    inpath : the path of csr format data(vlist elist info)
    outpath : the path of csc format data(vlist elist info)
    */
    if(argc != 3)
    {
        std::cout << "usage: ./csr2csc inpath outpath" << std::endl;
        return 0;
    }

    std::string inpath = argv[1];
    std::string outpath = argv[2];
    std::string vpath = inpath + "/vlist.bin";
    std::string epath = inpath + "/elist.bin";
    std::string ipath = inpath + "/info.bin";
    std::vector<int> vlist, elist, info;
    int v_cnt, e_cnt, n;

    v_cnt = read_binary2vector(vpath, vlist);
    e_cnt = read_binary2vector(epath, elist);
    n = read_binary2vector(ipath, info);
    int vertex_cnt = info[0];
    int rule_cnt = info[1];
    std::cout << "v_cnt: " << v_cnt << " e_cnt: " << e_cnt << std::endl;

    std::vector<int> vlist_csc(v_cnt, 0);
    std::vector<int> elist_csc(e_cnt, 0);
    std::vector<std::pair<int, int>> edges;
    for(int i = 0; i < v_cnt - 1; i ++)
    {
        for(int j = vlist[i]; j < vlist[i + 1]; j ++)
        {
            edges.emplace_back(std::make_pair(elist[j], i));
        }
    }
    std::sort(edges.begin(), edges.end());
    for(int i = 0; i < edges.size(); i ++)
    {
        vlist_csc[edges[i].first + 1] ++;
        elist_csc[i] = edges[i].second;
    }
    for(int i = 0; i < v_cnt; i ++)
    {
        vlist_csc[i + 1] += vlist_csc[i];
    }

    // write csc format to binary file
    std::string vpath_csc = outpath + "/vlist.bin";
    std::string epath_csc = outpath + "/elist.bin";
    std::string ipath_csc = outpath + "/info.bin";
    FILE* fvlist_csc = fopen(vpath_csc.c_str(), "w");
    FILE* felist_csc = fopen(epath_csc.c_str(), "w");
    FILE* finfo_csc = fopen(ipath_csc.c_str(), "w");
    fwrite(&vlist_csc[0], sizeof(int), vlist_csc.size(), fvlist_csc);
    fwrite(&elist_csc[0], sizeof(int), elist_csc.size(), felist_csc);
    fwrite(&vertex_cnt, sizeof(int), 1, finfo_csc);
    fwrite(&rule_cnt, sizeof(int), 1, finfo_csc);

    std::cout << "convert done !" << std::endl;
    return 0;
}