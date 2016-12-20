def remainFib(number):
    n = int(number)
    f0 = 1
    f1 = 1
    f = 0
    c = 0
    non_fib = 0
    while True:
        f = f0 + f1
        f0 = f1
        f1 = f
        d = f1 - f0 -1
        non_fib += d
        if non_fib >= n:
            old_non_fib = non_fib - d
            tmp_c = f0
            for _ in range(d+1):
                tmp_c += 1
                old_non_fib += 1
                if old_non_fib == n:
                    return str(tmp_c)
