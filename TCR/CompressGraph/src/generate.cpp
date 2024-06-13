#include <omp.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>
#include <fstream>
#include "../core/io.hpp"
#include "../core/util.hpp"

int getDepth(std::vector<int>& vlist, std::vector<int>& elist, std::vector<int>& depth, int length)
{
    
    int ans = 0;
    auto getOneDepth = [&](int v){
        std::queue<int> q;
        int depth = -1;
        q.push(v);
        while(q.size())
        {
            int sz = q.size();
            depth ++;
            while(sz --)
            {   
                int u = q.front();
                q.pop();
                // if(u < length) continue; 万恶之源
                for(int i = vlist[u]; i < vlist[u+1]; i++)
                {
                    int v = elist[i];
                    if(v >= length)
                    {
                        q.push(v);
                    }
                }
            }
        }
        return depth;
    };
    for(int i = 0; i < vlist.size() - 1; i ++)
    {
        int tmp = getOneDepth(i);
        // std::cout << "i: " << i << " depth: " << tmp << " v_list: " << vlist.size() << " depth_size: " << depth.size() <<std::endl;
        depth[i] = tmp;
        ans = std::max(ans, tmp);
    }
    return ans;
}

void write_data(std::vector<int>& vlist, std::vector<int>& elist, std::vector<int>& list, FILE* outfile)
{
    /* 
    write the coo format data to the outfile for every depth

    for efficiently reading, we write data in binary format
    */

    // record the first row and second row of coo format data respectively  
    std::vector<int> coo_v0, coo_v1;
    int v = 0, e = 0;
    for(int v0 : list)
    {
        v ++;
        e += vlist[v0 + 1] - vlist[v0];
        for(int v1 = vlist[v0]; v1 < vlist[v0 + 1]; v1 ++)
        {
            coo_v0.push_back(v0);
            coo_v1.push_back(elist[v1]);
        }
    }
    
    // write data 
    // for(int v0 : coo_v0)
    // {
    //     outfile << v0 << " ";
    // }
    // outfile << std::endl;
    // for(int v1 : coo_v1)
    // {
    //     outfile << v1 << " ";
    // }
    // outfile << std::endl;
    // std::cout << "v: " << v << " e: " << e << std::endl;

    // write data in binary format: length and content
    int len = coo_v0.size();
    fwrite(&len, sizeof(int), 1, outfile);
    fwrite(&coo_v0[0], sizeof(int), coo_v0.size(), outfile);
    fwrite(&coo_v1[0], sizeof(int), coo_v1.size(), outfile);
}
int main(int argc, char** argv)
{
    std::string pvlist = argv[1];
    std::string pelist = argv[2];
    std::string pinfo = argv[3];
    std::string outpath = argv[4];

    std::vector<int> vlist;
    std::vector<int> elist;
    std::vector<int> info;
    // read data for binary file
    int v_cnt = read_binary2vector(pvlist, vlist);
    int e_cnt = read_binary2vector(pelist, elist);
    std::cout << v_cnt << " " << e_cnt << std::endl;
    int n = read_binary2vector(pinfo, info);
    // v_cnt = vlist.size() = vertex_cnt + 1
    int vertex_cnt = info[0];
    int rule_cnt = info[1];
    std::cout << "vertex_cnt: " << vertex_cnt << " rule_cnt: " << rule_cnt << std::endl;

    // store depth for every vertex, included the rules
    std::vector<int> depth(v_cnt - 1, 0);

    // vertex_cnt is the real vertex number for origin graph
    int length = vertex_cnt;
    std::cout << "length: " << length << " depth_size: "<< depth.size() <<  std::endl;

    // get the depth for every vertex and store depth_list
    int max_depth = getDepth(vlist, elist, depth, length);
    std::cout << "max_depth: " << max_depth << std::endl;
    std::vector<std::vector<int>> depth_list(max_depth + 1);
    for(int i = 0; i < depth.size(); i ++)
    {
        depth_list[depth[i]].push_back(i);
    }
    std::cout << "generate depth_list done" << std::endl;

    std::cout << vlist[vlist.size() - 1] << " " << elist.size() << std::endl;
    for(int i = 0; i < depth_list.size(); i ++)
    {
        int e = 0;
        for(int j = 0; j < depth_list[i].size(); j ++)
        {
            e += vlist[depth_list[i][j] + 1] - vlist[depth_list[i][j]];
        }
        std::cout << "depth: " << i << " size: " << depth_list[i].size() << " e:" << e << std::endl;
    }
    // write data for every depth
    FILE* outfile = fopen(outpath.c_str(), "w");

    //write vertex_cnt, rule_cnt, max_depth
    fwrite(&vertex_cnt, sizeof(int), 1, outfile);
    fwrite(&rule_cnt, sizeof(int), 1, outfile);
    fwrite(&max_depth, sizeof(int), 1, outfile);
    // outfile << vertex_cnt << " " << rule_cnt << " " << max_depth <<std::endl;


    for(int i = 0; i < depth_list.size(); i ++)
    {
        // std::cout << "depth: " << i << " write data begin " <<std::endl;
        write_data(vlist, elist, depth_list[i], outfile);
        // std::cout << "depth: " << i << " write data done " <<std::endl;
    }

    fclose(outfile);

    int e = 0;
    for(int i = 0; i < depth_list.size(); i ++)
    {
        for(int j = 0; j < depth_list[i].size(); j ++)
        {
            e += vlist[depth_list[i][j] + 1] - vlist[depth_list[i][j]];
        }
    }
    std::cout << e << std::endl;
    return 0;
}