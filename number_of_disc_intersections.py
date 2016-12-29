# you can write to stdout for debugging purposes, e.g.
# print "this is a debug message"

def solution(A):
    n = len(A)
    count = 0
    layers = [0]*n
    current_layer = 0

    for i in range(n):
        b_r = i + A[i]
        b_l = i - A[i]
        if b_r >= n or b_r < i:
            b_r = n-1
        if b_l < 0:
            b_l = 0

        layers[b_r] += 1
        closing_layer = layers[i]
        layers[i] = current_layer

        if i > 0:
            if i - b_l > 1:
                count += i - b_l - 1
            count += layers[b_l]
            if count > 100000000:
                return -1

        layers[i] += 1
        current_layer += 1 - closing_layer

    return count
