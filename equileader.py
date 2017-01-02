#! /usr/bin/env python

def solution(A):

    h = {}
    for x in A:
        if x not in h:
            h[x] = 1
        else:
            h[x] += 1



    m1 = -float("inf")
    m2 = [x for x in h.keys() if h[x] == max(h.values())][0]

    equis = 0
    d = {}
    for k, x in enumerate(A):
        h[x] -= 1
        if x not in d:
            d[x] = 1
        else:
            d[x] += 1

        
