def polygonPlotOfLand(n, s, prices):
    import math
    if n == 0:
        return 0
    A = n * s * s / 4. / math.tan(math.pi/n)
    p = n * s

    cost_str = str(round(A * prices[0] + p * prices[1],2))
    res = cost_str.split('.')
    if len(res) > 1:
        if len(res[1]) < 2:
            cost_str = cost_str + '0'
    return cost_str
