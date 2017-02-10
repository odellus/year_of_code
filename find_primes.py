import math
# import numpy as np

def primes_under(N):
    if N == 1:
        return []
    if N == 2:
        return [2]
    if N == 3:
        return [2,3]

    sqrtN = int(round(math.sqrt(N)))
    primes = [2,3]
    for j in range(3, N, 2):
        founddivisor = False
        for p in primes:
            if p > sqrtN:
                break
            elif j % p == 0:
                founddivisor = True
                break
        if not founddivisor:
            primes.append(j)

    return primes

def is_prime(N):
    if N < 0 or type(N) != int:
        return False
    if N == 0 or N == 1:
        return False
    if N == 2 or N == 3:
        return True

    sqrtN = int(round(math.sqrt(N)))+1
    for j in range(2,sqrtN):
        # print(j)
        if N % j == 0:
            return False
    return True

def test_primes_under():
    primes_10 = primes_under(10)
    primes_100 = primes_under(100)
    primes_1000 = primes_under(1000)
    # print(primes_10)
    u = [is_prime(x) for x in primes_10]
    print(u)
    print([(x, is_prime(x)) for x in range(100)])
    is_prime(10)
    # print(primes_100)
    # print(primes_1000)

if __name__ == "__main__":
    test_primes_under()
