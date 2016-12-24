#! /usr/bin/python

def solution(A):

    def argmin(x):
        min_val, min_index = float("inf"), 0
        for k, val in enumerate(x):
            if val < min_val:
                min_val = val
                min_index = k
        return min_val, min_index

    avg_3_slice = []
    avg_2_slice = []

    n = len(A)

    for k in range(n-1):
        two_slice = (A[k] + A[k+1])/2.
        avg_2_slice.append(two_slice)
        if k < n-2:
            three_slice = (A[k] + A[k+1] + A[k+2])/3.
            avg_3_slice.append(three_slice)


    min_val3, min_index3 = argmin(avg_3_slice)
    min_val2, min_index2 = argmin(avg_2_slice)

    if min_val2 < min_val3:
        return min_index2
    elif min_val2 == min_val3:
        return min(min_index2, min_index3)
    else:
        return min_index3

if __name__ == "__main__":
    A = [21,62,33,84,15]
    B = range(20)
    solution(A)
    solution(B)
