// File: max_product_of_three.cpp

#include <algorithm>
#include <vector>
#define max(x, y) x > y ? x : y

int solution(vector<int> &A) {
    int n = A.size();
    std::sort(A.begin(), A.end());
    return max( A[n-1] * A[n-2] * A[n-3],
                A[0] * A[1] * A[n-1]);
}
