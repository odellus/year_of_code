import sys, copy
def solution(A):
    A.sort()
    ans1 = A[-1] * A[-2] * A[-3]
    ans2 = A[-1] * A[0] * A[1]
    return max(ans1, ans2)
