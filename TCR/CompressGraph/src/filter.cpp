#include <omp.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <tuple>
#include <vector>
#include <queue>

#include "../core/io.hpp"
#include "../core/util.hpp"

typedef struct rule_info { // rule infomation
    int len;
    int freq;
    rule_info() {
        len = 0;
        freq = 0;
    }
} rule_info;

int threshold = 0;

inline bool judgeRule(int len, int freq) {
    // 最差情况下 (freq - 1) * (len - 1) 的值为 -1
    // 即当 threshold == -2 时, 相当于没有进行压缩
    return (freq - 1) * (len - 1) - 1 <= threshold;
}

void mergeRule(VertexT rule_id, std::vector<std::vector<VertexT>> &graph,
               std::vector<std::vector<VertexT>> &graphT,
               std::vector<rule_info> &rif, std::vector<bool> &merge_flag,
               VertexT vertex_cnt, VertexT rule_cnt) {
    VertexT ID = rule_id + vertex_cnt;
    // insert childnode to parent node
    for (VertexT i = 0; i < graphT[ID].size(); ++i) {
        VertexT node = graphT[ID][i];
        std::vector<VertexT>::iterator it =
            find(graph[node].begin(), graph[node].end(), ID);
        std::vector<VertexT>::iterator pos = graph[node].erase(it);
        graph[node].insert(pos, graph[ID].begin(), graph[ID].end());
        // update len of parent node if parent node is rule
        if (node >= vertex_cnt) {
            rif[node - vertex_cnt].len += (graph[ID].size() - 1);
            merge_flag[node - vertex_cnt] = judgeRule(
                rif[node - vertex_cnt].len, rif[node - vertex_cnt].freq);
        }
    }
    // update child node idx
    for (VertexT i = 0; i < graph[ID].size(); ++i) {
        VertexT child_id = graph[ID][i];
        std::vector<VertexT>::iterator it =
            find(graphT[child_id].begin(), graphT[child_id].end(), ID);
        // delete origin idx in ruleID
        std::vector<VertexT>::iterator pos = graphT[child_id].erase(it);
        // insert new idx to child id
        graphT[child_id].insert(pos, graphT[ID].begin(), graphT[ID].end());
        // update freq of child node if the child node is rule
        if (child_id >= vertex_cnt) {
            rif[child_id - vertex_cnt].freq = graphT[child_id].size();
            merge_flag[child_id - vertex_cnt] =
                judgeRule(rif[child_id - vertex_cnt].len,
                          rif[child_id - vertex_cnt].freq);
        }
    }
}

void initInfoForRule(std::vector<std::vector<VertexT>> &graph,
                     std::vector<std::vector<VertexT>> &graphT,
                     std::vector<rule_info> &rif, std::vector<bool> &merge_flag,
                     VertexT vertex_cnt,
                     VertexT rule_cnt) { // init len&freq for each rule
    for (VertexT i = 0; i < rule_cnt; ++i) {
        rif[i].freq = graphT[i + vertex_cnt].size();
        rif[i].len = graph[i + vertex_cnt].size();
        merge_flag[i] = judgeRule(rif[i].len, rif[i].freq);
    }
}
int getDepth(std::vector<int>& out_vlist, std::vector<int>& out_elist, uint length)
{
    int res = 0;
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
                // if(u < length) continue;
                for(int i = out_vlist[u]; i < out_vlist[u+1]; i++)
                {
                    int v = out_elist[i];
                    if(v >= length)
                    {
                        q.push(v);
                    }
                }
            }
        }
        return depth;
    };

    for(int i = length; i < out_vlist.size(); i++)
    {
        int tmp = getOneDepth(i);
        res = std::max(tmp, res);
    }
    return res;
}
int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: filter <vlist> <elsit> <info> <threadshold>\n");
        return 0;
    }
    std::string pvlist = argv[1];
    std::string pelist = argv[2];
    std::string pinfo = argv[3];
    threshold = atoi(argv[4]);
    std::cout << "threshold: " << threshold << std::endl;
    std::vector<VertexT> vlist;
    std::vector<VertexT> elist;
    std::vector<VertexT> info;
    int v_cnt = read_binary2vector(pvlist, vlist);
    int e_cnt = read_binary2vector(pelist, elist);
    std::cout << "old_vlist: " << vlist.size() << " old_elist: " << elist.size() << std::endl;
    int n = read_binary2vector(pinfo, info);
    int vertex_cnt = info[0];
    int rule_cnt = info[1];
    std::cout << "vertex_cnt: " << vertex_cnt << "  old_rule_cnt: "<< rule_cnt << std::endl;
    fprintf(stderr, "filter start...\n");
    double start = timestamp();
    //VertexT -> int 
    std::vector<std::vector<VertexT>> graph(vertex_cnt + rule_cnt);
    csr_convert(graph, vlist, elist, vertex_cnt + rule_cnt);
    std::vector<std::vector<VertexT>> graphT(vertex_cnt + rule_cnt);
    csr_convert_idx(graphT, vlist, elist, vertex_cnt + rule_cnt);
    std::vector<rule_info> rif;
    rif.resize(rule_cnt);
    std::vector<bool> merge_flag(rule_cnt);
    initInfoForRule(graph, graphT, rif, merge_flag, vertex_cnt, rule_cnt);
    for (VertexT i = 0; i < rule_cnt; i++) {
        if (merge_flag[i] == true) {
            mergeRule(i, graph, graphT, rif, merge_flag, vertex_cnt, rule_cnt);
        }
    }
    // gen new ID for rule
    VertexT newRule_cnt;
    std::vector<VertexT> newRuleId(rule_cnt);
    genNewIdForRule(newRuleId, newRule_cnt, merge_flag, vertex_cnt, rule_cnt);
    std::vector<VertexT> new_vlist;
    std::vector<VertexT> new_elist;
    genNewGraphCSR(new_vlist, new_elist, graph, newRuleId, merge_flag,
                   vertex_cnt);
    double end = timestamp();
    fprintf(stderr, "Filter time : %.4f(s)\n", end - start);
    std::string outpath = argv[5];
    FILE *fvlist;
    FILE *felist;
    FILE *finfo;
    std::string vlist_path = outpath + "/vlist.bin";
    std::string elist_path = outpath + "/elist.bin";
    std::string info_path = outpath + "/info.bin";
    fvlist = fopen(vlist_path.c_str(), "w");
    felist = fopen(elist_path.c_str(), "w");
    fwrite(&new_vlist[0], sizeof(int), new_vlist.size(), fvlist);
    fwrite(&new_elist[0], sizeof(int), new_elist.size(), felist);
    finfo = fopen(info_path.c_str(), "w");
    fwrite(&vertex_cnt, sizeof(int), 1, finfo);
    fwrite(&newRule_cnt, sizeof(int), 1, finfo);
    // std::cout << vertex_cnt << " " << rule_cnt << std::endl;
    std::cout << "new_rule_cnt: " <<  newRule_cnt << "  new_vertex_cnt: " << vertex_cnt << std::endl;
    std::cout << "new_vlist: " << new_vlist.size() << " new_elist: " << new_elist.size() << std::endl;
    int length = vertex_cnt;
    int depth = getDepth(new_vlist, new_elist, length);
    std::cout << "final rule depth: " << depth << std::endl;
    
    double cr = double(vlist.size() + elist.size()) / (new_vlist.size() + new_elist.size());
    std::cout << "Compress ratio after filter: " << cr << std::endl;

    
    return 0;
}
