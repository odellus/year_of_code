# Author: Thomas Wood -- thomas@synpon

def solution(A):
    # write your code in Python 2.7
    import sys
    if len(A) == 2:
        return abs(A[0] - A[1])
    # Scan the array so you know what lies where in terms of sums.
    scan = []
    s = 0
    for x in A:
        s += x
        scan.append(s) #inclusive scan.

    mindiff = sys.maxint
    for x in scan[:-1]:
        diff = abs(s - 2*x)
        if diff < mindiff:
            mindiff = diff

    return mindiff
