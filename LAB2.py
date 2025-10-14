import numpy as np

# a = np.arange(1, 10, 2)
#
# print(a)
#
# print(a.dtype)


# a = np.array([[1,2,3],[4,5,6]])
#
# print(a)
# print(a.shape)
# print(a.size)

# a = np.array([1,2,3,4,5,6,7,8])
#
# print(a)
#
# print(a.shape)
# print(a.size)
#
# a = a.reshape((2,4))
#
# print(a)
# print(a.shape)



# a = np.array([[1,2,3],[4,5,6]])
#
# print("Array = ",a)
# print("Shape = ",a.shape)
# print("Dim = ",a.ndim)
# print("Size = ",a.size)
# print("Type = ",a.dtype)
# print("ItemSize = ",a.itemsize)



# a = np.array([1,2,3,4,5,6])
#
# print("a[0] = ",a[0])
# print("a[-1] = ",a[-1])
# print("a[1:3] = ",a[1:3])
# print("a[:2] = ",a[:2])


# a = np.array([[1,2,3],[4,5,6]])
#
# print("a[0] = ",a[0])
#
# print("a[-1] = ",a[-1])
#
# print("a[1:4] = ",a[1:4])
#
# print("a[:2] = ",a[:2])


# a = np.array([[1,2,3,4,5,6],[7,8,9,10,11,12]])
#
# print("a[a>5] = ", a[a>5])
#
# print("a[a%2==0] = ", a[a%2==0])
#
# for i in a:
#     print(i)



# v = np.array([1,2,3])
#
# u = np.array([10,20,30])
#
# print("sum = ",v+u)
#
# print("dif = ",u-v)
#
# print("mul = ",v*u)
#
# print("div = ",u/v)


# v = np.array([[1,2,3],[4,5,6]])
#
# u = np.array([[10,20,30],[40,50,60]])
#
# v += u
#
# print("sum = ",v)


# m = np.array([[1,2,3],[4,5,6]])
#
# print("add = ",m+2)
#
# print("sub = ",m-1)
#
# print("mul = ",m*2)
#
# print("div = ",m/2)



# v = np.array([[1,2,3],[4,5,6]])
#
# u = np.array([4,5,6])
#
# print("dot = ",v.dot(u))


# m = np.array([[1,2,3],[4,5,6]])
#
# print("matrix = ",m)
#
# print("transpose = ",m.transpose())


# m = np.array([[1,2,3],[4,5,6]])
#
# print("sum = ",m.sum())
#
# print("min = ",m.min())
#
# print("max = ",m.max())
#
# print("mean = ",m.mean())



# m = np.array([[1,2,3],[4,5,6]])
#
# print("sum = ",m.sum(axis=0))
#
# print("min = ",m.min(axis=0))
#
# print("max = ",m.max(axis=0))
#
# print("mean = ",m.mean(axis=0))


v = np.array([1,3,5,7,9])

np.save("my_array", v)

u = np.load("my_array.npy")

print(u)