import numpy as np
l = [2, 4, 0, 100, 4, 11, 2602, 36]
li = np.array(l)
li = li%2
print(li)
print(len(np.argwhere(li==1)))
# print(np.argwhere(li==1)[0])
# print(l[5])
# print(l[int(np.argwhere(li==1)[0])])
