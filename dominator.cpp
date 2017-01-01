#include <vector>
#include <unordered_map>
#include <limits>

int solution(vector<int> &A) {
    int n = A.size();
    float beat_that = n / 2;
    unordered_map<int, int> h;

    for (int i=0; i < n; i++) {
        if (h.find(A[i]) == h.end()) {
            h.insert(pair<int,int> (A[i], 1));
        }
        else {
            h[A[i]] += 1;
        }
    }

    int max_occur = std::numeric_limits<int>::infinity();
    int max_val = -1;

    for (auto it = h.begin(); it != h.end(); ++it) {
        if (it->second > max_occur) {
            max_occur = it->second;
            max_val = it->first;
        }
    }

    int max_index = -1;
    for (int i=0; i < n; i++) {
        if (A[i] == max_val) {
            max_index = i;
            break;
        }
    }
    int res = -1;
    if (float(max_occur) > beat_that) {
        res = max_index;
    }
    return res;

}
