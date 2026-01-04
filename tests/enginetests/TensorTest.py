import unittest
from src.engine.Tensor import Tensor, Operation, OpBackward

class TestTensor(unittest.TestCase):
    
    #addition
    def test_addition_data(self):
        # simple addition
        a = Tensor(10)
        b = Tensor(20)
        c = a + b
        self.assertEqual(c.data, 30)

    def test_graph_structure(self):
        # computation graph 
        a = Tensor(1)
        b = Tensor(2)
        c = a + b
        
        # c should have a grad_fn, but a and b (leaf nodes) should not
        self.assertIsNotNone(c.grad_fn)
        self.assertIsNone(a.grad_fn)
        self.assertIsNone(b.grad_fn)
        
        # check if Operation is correct
        self.assertEqual(c.grad_fn.operation, Operation.ADD)
        
        # check if parents are tracked correctly
        self.assertIn(a, c.grad_fn.parents)
        self.assertIn(b, c.grad_fn.parents)

    def test_chained_operations(self):
        # check if nested operations create a deeper graph
        a = Tensor(1)
        b = Tensor(2)
        c = Tensor(3)
        
        d = a + b
        e = d + c
        
        # e's parents should be d and c
        self.assertEqual(len(e.grad_fn.parents), 2)
        self.assertIn(d, e.grad_fn.parents)
        self.assertIn(c, e.grad_fn.parents)

    def test_int_plus_tensor(self):
        """Test: int + Tensor (uses __radd__)"""
        a = 5
        b = Tensor(10)
        c = a + b
        
        self.assertEqual(c.data, 15)
        self.assertEqual(c.grad_fn.operation, Operation.ADD)
        # Check that the first parent is the wrapped integer
        self.assertEqual(c.grad_fn.parents[0].data, 5)

    def test_graph_integrity(self):
        """Ensure the graph structure holds for scalar ops"""
        a = Tensor(1)
        b = 2
        c = a + b
        
        # This checks that we don't get the AttributeError
        # by checking if the parents in the grad_fn are all Tensors.
        for parent in c.grad_fn.parents:
            self.assertTrue(hasattr(parent, 'data'))
            self.assertTrue(hasattr(parent, 'grad_fn'))

    # multiplication
    def test_multiplication_data(self):
        """Simple multiplication: 10 * 20 = 200"""
        a = Tensor(10)
        b = Tensor(20)
        c = a * b
        self.assertEqual(c.data, 200)
        self.assertEqual(c.grad_fn.operation, Operation.MUL)

    def test_tensor_mul_int(self):
        """Test: Tensor * int (uses __mul__)"""
        a = Tensor(10)
        b = 3
        c = a * b
        self.assertEqual(c.data, 30)
        self.assertEqual(c.grad_fn.parents[0].data, 10)
        self.assertEqual(c.grad_fn.parents[1].data, 3)

    def test_int_mul_tensor(self):
        """Test: int * Tensor (uses __rmul__)"""
        a = 4
        b = Tensor(10)
        c = a * b
        self.assertEqual(c.data, 40)
        # Verify the scalar 4 is parent[0] because of your clean __rmul__ fix
        self.assertEqual(c.grad_fn.parents[0].data, 4)
        self.assertEqual(c.grad_fn.parents[1].data, 10)

    def test_mixed_ops_graph(self):
        """Test: (a * b) + c to ensure the graph handles mixed ops"""
        a = Tensor(2)
        b = Tensor(3)
        c = Tensor(4)
        
        # d = 2 * 3 = 6
        d = a * b
        # e = 6 + 4 = 10
        e = d + c
        
        self.assertEqual(e.data, 10)
        self.assertEqual(e.grad_fn.operation, Operation.ADD)
        self.assertEqual(e.grad_fn.parents[0].grad_fn.operation, Operation.MUL)

if __name__ == "__main__":
    unittest.main()