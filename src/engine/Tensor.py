from enum import Enum

class Operation(Enum):
	ADD = "add"
	MUL = "mul"

class OpBackward():
	def __init__(self, operation, parents):
		self.operation = operation
		self.parents = parents

class Tensor():
	def __init__(self, data):
		self.data = data
		self.grad = None
		self.grad_fn = None
	
	def backward(self):
		pass

	def __add__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)

		res = Tensor(self.data + other.data)
		res.grad_fn = OpBackward(Operation.ADD, [self, other])

		return res

	def __radd__(self, other):
		return Tensor(other).__add__(self)

	def __mul__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)

		res = Tensor(self.data * other.data)
		res.grad_fn = OpBackward(Operation.MUL, [self, other])

		return res

	def __rmul__(self, other):
		return Tensor(other).__mul__(self)
'''

a = Tensor(1)
b = Tensor(2)

c = a + b

print(f"a_gradfn: {a.grad_fn}, b_gradfn: {b.grad_fn}, c_gradfn: {c.grad_fn}")


'''

