# Author: Thomas Wood -- thomas@synpon.com

def solution(X, A):
    h = {}
    leaf_counter = 0
    for k, leaf_pos in enumerate(A):
        if leaf_pos not in h:
            h[leaf_pos] = 1
            leaf_counter += 1
        if leaf_counter == X:
            # All the spots are covered.
            return k
    # Went through whole array without covering 1 through X.
    return -1
