def sequence(upper_bound):
    res = []
    diff = 1
    x = 1
    while x <= upper_bound:
        res.append(x)
        x += diff
        #diff = 3 if diff == 1 else 1
    return res  #', '.join(res)

a = sequence(10)
print(a)
