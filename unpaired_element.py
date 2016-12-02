def solution(A):
    unpaired = 0
    for x in A:
        unpaired ^= x

    return unpaired
