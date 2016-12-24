// File: count_div.cpp
// Author: Thomas Wood -- thomas@synpon.com
int solution(int A, int B, int K) {
    int res = 0;
    int rA = A % K;
    int rB = A % K;
    int lA = A - rA;
    int lB = B - rB;
    if (rA == 0 && rB == 0) {
        res = (B - A) / K + 1;
    }
    else if (rA == 0 && rB != 0) {
        if (lB >= A) {
            res = (lB - A) / K + 1;
        }
        else {
            res = 0;
        }
    }
    else if (rA != 0 && rB != 0) {
        if (lB >= A) {
            res = (lB - lA) / K;
        }
        else {
            res = 0;
        }
    }
    else {
        res = (B - lA) / K;
    }

    if (res < 1) {
        res = 0;
    }

    return res;
}
