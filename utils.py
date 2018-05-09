from math import log

def sum_weights(w1, w2):
    a = max(w1, w2)
    b = min(w1, w2)
    return a + log(1 + 2**(b-a))/log(2)

