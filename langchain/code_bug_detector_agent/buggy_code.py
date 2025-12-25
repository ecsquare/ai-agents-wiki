"""
Test file with intentional bugs for the Code Bug Fixer
Save this as 'buggy_code.py' and run the fixer on it
"""

import os
import sys
import json
import math
import random


# Bug 1: Comparison instead of assignment
def calculate_total(items):
    total == 0  # Should be = not ==
    for item in items:
        total += item['price']
    return total


# Bug 2: Empty except block
def safe_divide(a, b):
    try:
        return a / b
    except:
        pass  # Should handle the exception properly


# Bug 3: Mutable default argument
def add_to_list(item, my_list=[]):
    my_list.append(item)
    return my_list


# Bug 4: Missing return statement
def get_username(user_id):
    if user_id == 1:
        return "admin"
    elif user_id == 2:
        return "user"
    # Missing else case or default return


# Bug 5: Potential undefined variable
def process_order(order):
    if order['status'] == 'pending':
        result = "Processing order"

    return result  # result might not be defined


# Bug 6: Unused imports (random is imported but not used)
def calculate_area(radius):
    return math.pi * radius ** 2


# Bug 7: Syntax error (missing colon)
def broken_function(x)  # Missing colon
    return x * 2


# Bug 8: Indentation error
def another_function():
    result = 10  # Wrong indentation


return result

# Main code
if __name__ == "__main__":
    items = [
        {"name": "Widget", "price": 10.99},
        {"name": "Gadget", "price": 25.50}
    ]

    total = calculate_total(items)
    print(f"Total: ${total}")