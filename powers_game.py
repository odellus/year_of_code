#! /usr/bin/env python3
import sys

T = int(input().strip())
for a0 in range(T):
    n = int(input().strip())
    if n % 8 == 0:
        print("Second")
    else:
        print("First")
