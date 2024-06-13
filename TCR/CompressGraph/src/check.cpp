#include <iostream>
#include <fstream>

#include "../core/compression/basics.h"
#include "../core/compression/hash.h"
#include "../core/compression/heap.h"
#include "../core/compression/records.h"
#include "../core/io.hpp"
#include "../core/util.hpp"

int main()
{
    std::string pvlist = "/root/Project/CompressGraph/dataset/cnr-2000/origin/csr_vlist.bin";
    std::string pelist = "/root/Project/CompressGraph/dataset/cnr-2000/origin/csr_elist.bin";
    std::vector<int> csr_vlist, csr_elist;
    int v_cnt = read_binary2vector(pvlist, csr_vlist);
    int e_cnt = read_binary2vector(pelist, csr_elist);
    std::cout << "v_cnt: " << v_cnt << " e_cnt: " << e_cnt << std::endl;

    std::cout << csr_vlist[325556] << " " << csr_vlist[325557]<< std::endl;
    int max = 0;
    for(int i : csr_elist)
    {
        max = std::max(max, i);
    }
    std::cout << "max: " << max << std::endl;

}