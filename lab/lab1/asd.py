import numpy as np
re = [1,0,1,0,1,1,1,1]
lir = [x for x in range(2, 10000000)]
lir.reverse()
if not all(i == 1 or i == 0 for i in re):
    re2 = []
    for z in re:
        if z > 1:
            re2.append(z)
    d = np.min(re2)
    print(d)
d = lir.pop()
print(d)


