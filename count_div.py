#! /usr/bin/python

def solution(A, B, K):
    res = 0
    rem_A = A % K
    rem_B = B % K
    if rem_A == 0 and rem_B == 0:
        res = (B - A) / K + 1
    elif rem_A == 0 and rem_B != 0:
        low_B = B - rem_B
        if low_B >= A:
            res = (low_B - A) / K + 1
        else:
            res = 0
    elif rem_A != 0 and rem_B != 0:
        low_A  = A - rem_A
        low_B = B - rem_B
        if low_B >= A:
            res = (low_B - low_A) / K
        else:
            res = 0
    elif rem_A != 0 and rem_B == 0:
        low_A = A - rem_A
        res = (B - low_A) / K

    if res < 1:
        res = 0

    return res
