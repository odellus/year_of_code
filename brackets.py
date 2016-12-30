# you can write to stdout for debugging purposes, e.g.
# print "this is a debug message"

def solution(S):
    # write your code in Python 2.7
    stack = []
    n = len(S)

    for i in range(n):
        x = S[i]
        if x == '(' or x == '{' or x == '[':
            stack.append(x)
        elif len(stack) < 1:
            return 0
        else:
            last = stack.pop(-1)
            if not ((last == '(' and x == ')') or (last == '{' and x == '}') or (last == '[' and x == ']')):
                return 0

    if len(stack) < 1:
        return 1
    else:
        return 0
