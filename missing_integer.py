# Author : Thomas Wood -- thomas@synpon.com

def solution(A):
    # write your code in Python 2.7
    if len(A) == 1:
        if A[0] == 1:
            return 2
        else:
            return 1
    h = {}
    for x in A:
        # Keep track of positive integers in A.
        if x > 0 and x not in h:
            h[x] = 1

    # All negative values.
    if len(h.keys()) < 1:
        return 1
    m = min(h.keys())
    if m > 1:
        return 1
    k = m + 1
    while k in h:
        k += 1

    return k
