def CodeFight(n):
    res = []
    for k in range(1,n+1):
        if k % 5 == 0 and k % 7 != 0:
            res.append("Code")
        elif k % 5 != 0 and k % 7 == 0:
            res.append("Fight")
        elif k % 35 == 0:
            res.append("CodeFight")
        else:
            res.append(str(k))
    return res
