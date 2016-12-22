#! /usr/bin/env/ python
# -*- coding: utf-8

def solution(A):
    # write your code in Python 2.7
    scan = []
    s = 0
    if len(A) < 2:
        return 0
    for x in A:
        s += x
        scan.append(s)

    total = scan[-1]

    passing = 0
    for k, x in enumerate(scan):
        if A[k] == 0:
            passing += total - x


    if passing <= 1e9:
        return passing
    else:
        return -1
