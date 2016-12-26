// File: max_product_of_three.cpp

#include <algorithm>
#include <vector>
#define max(x, y) x > y ? x : y

int solution(vector<int> &A) {
    std::sort(A.begin(), A.end());
    return max( A[A.end()] * A[A.end()-1] * A[A.end()-2],
                A[0] * A[1] * A[A.end()]))
}
