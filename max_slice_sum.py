def solution(A):
    import sys
    if len(A) < 1:
        return 0
    elif len(A) == 1:
        return A[0]

    if len(A) == 2:
        a1, a2 = A
        a3 = sum(A)
        return max(a1,a2, a3)

    if len(A) == 3:
        a1, a2, a3 = A
        a4 = a1 + a2
        a6 = a2 + a3
        a7 = sum(A)
        return max(a1,a2,a3,a4,a6,a7)

    s = 0
    m = -sys.maxint
    for x in A:
        s += x

        if s < 0:
            s = x

        if s > m:
            m = s

        if x > m:
            m = x
    return m
