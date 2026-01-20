from enum import Enum

class Operation(Enum):
	ADD = "add"
	MUL = "mul"
	SUB = "sub"
	RELU = "relu"
	#MATMUL = "matmul" #matmul, pow, truediv tbd later. for now, a scalar babygrad it is.
	SUM = "sum"


class OpBackward():
	def __init__(self, operation, parents):
		self.operation = operation
		self.parents = parents

class Tensor():
	def __init__(self, data, requires_grad=False):
		self.data = data
		self.requires_grad = requires_grad
		self.grad = 0.0 if requires_grad else None
		self.grad_fn = None
	
	def backward(self):
		pass

	def _build_toplogical_ordering(self):
		topo_seq = []
		visited = set() #made it set only for O(1) lookup perf

		#for an operand node,
		def traversal(node):
			if node not in visited:
				visited.add(node)
				if node.grad_fn:
					for parent in node.grad_fn.parents:
						traversal(parent)
				topo_seq.append(node)

		traversal(self)
		topo_seq = list(reversed(topo_seq))
		return topo_seq

	def zero_grad(self):
		topo_seq = self._build_toplogical_ordering()
		for node in topo_seq:
			if node.grad is not None:
				node.grad = 0.0

	def __add__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		
		requires_grad = self.requires_grad or other.requires_grad

		res = Tensor(self.data + other.data, requires_grad=requires_grad)
		res.grad_fn = OpBackward(Operation.ADD, [self, other]) if requires_grad else None

		return res

	def __radd__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		
		requires_grad = other.requires_grad or self.requires_grad

		res = Tensor(other.data + self.data, requires_grad=requires_grad)
		res.grad_fn = OpBackward(Operation.ADD, [other, self]) if requires_grad else None

		return res

	def __mul__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		
		requires_grad = self.requires_grad or other.requires_grad

		res = Tensor(self.data * other.data, requires_grad=requires_grad)
		res.grad_fn = OpBackward(Operation.MUL, [self, other]) if requires_grad else None

		return res

	def __rmul__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
		
		requires_grad = other.requires_grad or self.requires_grad

		res = Tensor(other.data * self.data, requires_grad=requires_grad)
		res.grad_fn = OpBackward(Operation.MUL, [other, self]) if requires_grad else None

		return res

	def __sub__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)
			
		requires_grad = self.requires_grad or other.requires_grad

		res = Tensor(self.data - other.data, requires_grad=requires_grad)
		res.grad_fn = OpBackward(Operation.SUB, [self, other]) if requires_grad else None

		return res

	def __rsub__(self, other):
		if not isinstance(other, Tensor):
			other = Tensor(other)

		requires_grad = self.requires_grad or other.requires_grad

		res = Tensor(other.data - self.data, requires_grad=requires_grad)
		res.grad_fn = OpBackward(Operation.SUB, [other, self]) if requires_grad else None

		return res

	def relu(self):
		res = self.data if self.data > 0 else 0
		res = Tensor(res, requires_grad=self.requires_grad)
		res.grad_fn = OpBackward(Operation.RELU, [self]) if self.requires_grad else None

		return res

