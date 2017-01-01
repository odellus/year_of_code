# you can write to stdout for debugging purposes, e.g.
# print "this is a debug message"

def solution(A):
    # write your code in Python 2.7
    n = len(A)
    beat_that = n / 2.0

    h = {}
    for x in A:
        if x not in h:
            h[x] = 1
        else:
            h[x] += 1

    max_occur = -float("inf")
    max_val = -1
    for x in h.keys():
        if h[x] > max_occur:
            max_occur = h[x]
            max_val = x

    for k in range(n):
        if A[k] == max_val:
            break


    if max_occur > beat_that:
        return k
    else:
        return -1
