
def naive_solution(N, A):
    # Goal is to calculate the value of every counter after all operations.
    counter = [0] * N
    for x in A:
        if 1 <= x and x <= N:
            counter[x] += 1
        elif x == N + 1:
            counter = [max(counter) for _ in counter] # O(MN) for worst case.

    return counter

def lazy_solution(N, A):
    counter = [0] * N
    max_counter = 0
    local_max = 0

    for x in A:

        if 1 <= x and x <= N:
            # increment(X)
            if counter[x-1] < max_counter:
                # if the value of counter[x-1] is less than the running max,
                # it hasn't been updated since before. Fix that.
                counter[x-1] = max_counter
            # Always increment the x-1 element of coutner.
            counter[x-1] += 1
            if local_max < counter[x-1]:
                # We're keeping track of local max so we can avoid O(N) operations
                # inside the loop.
                local_max = counter[x-1]

        else:
            max_counter = local_max

    # Now go through and check that all values are > max_counter. Fix if needed.
    for k in range(N):
        if counter[k] < max_counter:
            counter[k] = max_counter

    return counter
