# Author: Thomas Wood, thomas@synpon.com

def solution(A):
    N = len(A)
    if N == 0:
        return 1
    if N == 1:
        if A[0] == 1:
            return 2
        else:
            return 1

    u = [x for x in range(min(A), max(A)+1)]
    res = sum(u) - sum(A)
    if res == 0:
        if min(A) == 1:
            return N + 1
        else:
            return 1
    else:
        return res
