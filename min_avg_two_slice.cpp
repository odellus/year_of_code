// File: min_avg_two_sclie.cpp
#include <vector>
#include <limits>

#define min(x,y) x > y ? y : x;

struct ArgminResponse {
    float val;
    int index;
};

ArgminResponse argmin(vector<float> &A) {
    float min_val = std::numeric_limits<float>::infinity();
    int min_index = 0;

    for (int i=0; i < A.size(); i++) {
        if (A[i] < min_val) {
            min_val = A[i];
            min_index = i;
        }
    }
    ArgminResponse res;
    res.val = min_val;
    res.index = min_index;

    return res;
}

int solution(vector<int> &A) {

    int n = A.size();

    vector<float> avg_2_slice;
    vector<float> avg_3_slice;

    float two_slice;
    float three_slice;

    for (int i=0; i < n-1; i++) {
        two_slice = float(A[i] + A[i+1]) / 2;
        avg_2_slice.push_back(two_slice);
        if (i < n-2) {
            three_slice = float(A[i] + A[i+1] + A[i+2]) / 3.0;
            avg_3_slice.push_back(three_slice);
        }
    }

    ArgminResponse min2;
    ArgminResponse min3;

    min2 = argmin(avg_2_slice);
    min3 = argmin(avg_3_slice);
    // std::cout << min3.val << ' ' << min2.val << std::endl;

    if (min2.val < min3.val) {
        return min2.index;
    }
    else if (min2.val == min3.val) {
        return min(min2.index, min3.index);
    }
    else {
        return min3.index;
    }
}
