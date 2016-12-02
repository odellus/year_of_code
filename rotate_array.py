

def solution(A, K):
    # write your code in Python 2.7
    N = len(A)
    if N == 0:
        return A
    u = (K % N)
    if u == 0:
        return A
    else:
        return A[-u:] + A[:N-u]
