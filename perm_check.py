# Author : Thomas Wood -- thomas@synpon.com

def solution(A):
    if min(A) != 1 or max(A) != len(A):
        return 0

    h = {}
    for x in A:
        if x not in h:
            h[x] = 1
        else:
            return 0

    return 1
