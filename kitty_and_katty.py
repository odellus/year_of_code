#! /usr/bin/env python3

import sys


T = int(input().strip())
for a0 in range(T):
    n = int(input().strip())
    if n == 1:
        print("Kitty")
    elif n % 2 == 0:
        print("Kitty")
    else:
        print("Katty")
