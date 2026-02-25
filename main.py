import tensorx
print(tensorx)

import tensorx._tensorx
print(tensorx._tensorx)

print("add_ints:", tensorx.add_ints(2, 3))

print("make_tensor:")
tensorx.make_tensor()

t0 = tensorx.Tensor()
print("t0.rank:", t0.rank, "t0.dims:", t0.dims, "t0.length:", t0.length)
print("t0:", t0)

s = tensorx.scalar(3.14)
print("s:", s)
print("s.rank:", s.rank, "s.dims:", s.dims, "s.length:", s.length)
print("s.get:", s.get())     # 3.14
s.set(2.71)
print("s.set(2.71)->s.get:", s.get())     # 2.71

t1 = tensorx.Tensor(2, (2, 3))
print("t1.rank:", t1.rank, "t1.dims:", t1.dims, "t1.length:", t1.length)
print("t1:", t1)

t2 = tensorx.Tensor(2, 3, 4)
print("t2.rank:", t2.rank, "t2.dims:", t2.dims, "t2.length:", t2.length)
print("t2:", t2)

v = tensorx.from_list([1, 2, 3])
print("v:", v)

m = tensorx.from_list([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print("m:", m)
print("m.rank:", m.rank, "m.dims:", m.dims, "m.length:", m.length)
print("m.get(1, 1):", m.get(1, 1))
