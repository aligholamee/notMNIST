# Numerical Stability Test
# ========================================
# [] File Name : ns.py
#
# [] Creation Date : December 2017
#
# [] Created By : Ali Gholami (aligholami7596@gmail.com)
# ========================================
#
x = 1000000000
x += 0.000001
x -= 1000000000
print(x)
