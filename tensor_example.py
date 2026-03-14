import tensorxx
print(tensorxx)

import tensorxx._tensorxx
print(tensorxx._tensorxx)

print("add_ints:", tensorxx.add_ints(2, 3))

print("make_tensor:")
tensorxx.make_tensor()

t0 = tensorxx.Tensor()
print("t0.rank:", t0.rank, "t0.dims:", t0.dims, "t0.length:", t0.length)
print("t0:", t0)

s = tensorxx.scalar(3.14)
print("s:", s)
print("s.rank:", s.rank, "s.dims:", s.dims, "s.length:", s.length)
print("s.get:", s.get())     # 3.14
s.set(2.71)
print("s.set(2.71)->s.get:", s.get())     # 2.71

t1 = tensorxx.Tensor(2, (2, 3))
print("t1.rank:", t1.rank, "t1.dims:", t1.dims, "t1.length:", t1.length)
print("t1:", t1)

t2 = tensorxx.Tensor(2, 3, 4)
print("t2.rank:", t2.rank, "t2.dims:", t2.dims, "t2.length:", t2.length)
print("t2:", t2)

v = tensorxx.from_list([1, 2, 3])
print("v:", v)

m = tensorxx.from_list([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print("m:", m)
print("m.rank:", m.rank, "m.dims:", m.dims, "m.length:", m.length)
print("m.get(1, 1):", m.get(1, 1))
