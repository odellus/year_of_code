// File: passing_cars.cpp
// Author: Thomas Wood -- thomas@synpon.com

int solution(vector<int> &A) {
    // write your code in C++14 (g++ 6.2.0)
    int n = A.size();
    vector<int> scan;

    int s = 0;
    for (int j=0; j < n; j++) {
        scan.push_back(s);
        s += A[j];
    }

    unsigned int passing = 0;
    for (int j=0; j < n; j++) {
        if (A[j] == 0) {
            passing += s - scan[j];
        }
    }
    if (passing <= 1000000000) {
        return passing;
    }
    else {
        return -1;
    }
}
