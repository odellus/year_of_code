// File: distinct.cpp
#include <unordered_map>
#include <vector>

using namespace std;

int solution(vector<int> &A) {
    int n = A.size();
    unordered_map<int, int> map;
    int res = 0;
    int key;

    for (int i=0; i < n; i++) {
        if (map.find(A[i]) == map.end()) {
            map.insert(pair<int,int> (A[i], 1));
            res += 1;
        }
    }
    return res;
}
