# you can write to stdout for debugging purposes, e.g.
# print "this is a debug message"

def solution(A, B):
    pack = []
    eaten = 0
    n = len(A)
    assert len(A) == len(B)
    for i in range(n):
        if B[i] == 1:
            pack.append(A[i])
        elif B[i] == 0 and len(pack) > 0:
            while True:
                # The fish swimming upstream ate the pack
                if len(pack) < 1:
                    break
                elif A[i] < pack[-1]:
                    # The fish swimming upstream was eaten by one of the pack.
                    eaten += 1
                    # The pack continues downstream.
                    break
                elif A[i] > pack[-1]:
                    # Fish swimming upstream ate the leader of the pack.
                    eaten += 1
                    pack.pop(-1)

    return n-eaten
