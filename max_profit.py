def solution(A):
    if len(A) < 1:
        return 0

    m = 0
    b = A[0]
    for x in A:
        m = max(m, x - b)
        if x - b <= 0:
            b = x
    return m
