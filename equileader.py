#! /usr/bin/env python

def solution(A):
    n = len(A)
    h = {}
    for x in A:
        if x not in h:
            h[x] = 1
        else:
            h[x] += 1



    m1_occur = -float("inf")
    m1_val = -1
    m2_occur = max(h.values())
    m2_val = [x for x in h.keys() if h[x] == m2_occur][0]

    equis = 0
    d = {}
    for k, x in enumerate(A):
        h[x] -= 1
        if x not in d:
            d[x] = 1
        else:
            d[x] += 1
        if d[x] > m1_occur:
            m1_occur = d[x]
            m1_val = x

        if m1_occur > (k+1) / 2.0 and h[m2_val] > (n-1-k)/2.0 and m2_val == m1_val:
            equis += 1

    return equis
