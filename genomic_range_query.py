#! /usr/bin/env python

def solution(A, P, Q):
    h = {}
    n = len(A)
    m = len(P)
    res = []
    for k in range(m):
        i, j = P[k], Q[k]
        cut = A[i:j+1]
        key = str(i) + '_' + str(j)
        # We haven't seen these endpoints yet.
        if key not in h:
            if 'A' in cut:
                h[key] = 1
                res.append(h[key])
                continue
            if 'C' in cut:
                h[key] = 2
                res.append(h[key])
                continue
            if 'G' in cut:
                h[key] = 3
                res.append(h[key])
                continue
            if 'T' in cut:
                h[key] = 4
                res.append(h[key])
                continue
        else:
            res.append(h[key])


    return res
