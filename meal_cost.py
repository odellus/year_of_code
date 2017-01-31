#! /usr/bin/env python3

mealCost = float(input().strip())
tip = int(input().strip())
tax = int(input().strip())

totalCost = (1 + tip/100. + tax/100.) * mealCost

print("The total meal cost is {} dollars.".format(int(round(totalCost))))
