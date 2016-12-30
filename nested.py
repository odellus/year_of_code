#! /usr/bin/env python

def solution(A):
    st = []
    for ch in S:
        if ch == '(':
            st.append(ch)
        elif len(st) < 1:
            return 0
        elif st.pop(-1) != '(' or ch != ')':
            return 0

    if len(st) == 0:
        return 1
    else:
        return 0
