#! /usr/bin/env python

def solution(A):
    A.sort()
    n = len(A)
    for j in range(0,n-2):
        p, q, r = j, j+1, j+2
        if A[p] + A[q] > A[r] \
        and A[q] + A[r] > A[p] \
        and A[r] + A[p] > A[q]:
            return 1
    return 0
