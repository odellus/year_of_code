// File: genomic_range_query.cpp
#include <vector>
#include <string>
#include <unordered_map>


vector<int> solution(string &A, vector<int> &P, vector<int> &Q) {

    unordered_map<string, int> h;
    int n = A.size();
    int m = P.size();
    vector<int> res;

    int j, k;
    for (int i=0; i < m; i++) {
        j = P[i];
        k = Q[i];
        string cut = A.substr(j,k-j+1);

        string key = to_string(j) + "_" + to_string(k);
        if (h.find(key) == h.end()) {
            if (cut.find_first_of('A') != string::npos) {
                h.insert(std::pair<string,int>(key, 1));
                res.push_back(h[key]);
                continue;
            }
            if (cut.find_first_of('C') != string::npos) {
                h.insert(std::pair<string,int>(key, 2));
                res.push_back(h[key]);
                continue;
            }
            if (cut.find_first_of('G') != string::npos) {
                h.insert(std::pair<string,int>(key, 3));
                res.push_back(h[key]);
                continue;
            }
            if (cut.find_first_of('T') != string::npos) {
                h.insert(std::pair<string,int>(key, 4));
                res.push_back(h[key]);
                continue;
            }
        }
        else {
            res.push_back(h[key]);
        }
    }
    return res;

}
