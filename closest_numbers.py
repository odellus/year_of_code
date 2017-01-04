def closestNumbers(numbers):
    numbers.sort()
    mindiff = sys.maxint
    n = len(numbers)
    for k in range(n-1):
        absdiff = abs(numbers[k] - numbers[k+1])
        if absdiff < mindiff:
            mindiff = absdiff
    for k in range(n-1):
        absdiff = abs(numbers[k] - numbers[k+1])
        if absdiff == mindiff:
            print "{0} {1}".format(numbers[k],numbers[k+1])
