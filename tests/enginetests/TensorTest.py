import unittest
from src.engine.Tensor import Tensor, Operation

class TestTensor(unittest.TestCase):
    
    # --- Addition Tests ---
    def test_addition_data(self):
        # simple addition
        a = Tensor(10, requires_grad=True)
        b = Tensor(20, requires_grad=True)
        c = a + b
        self.assertEqual(c.data, 30)

    def test_graph_structure(self):
        # computation graph 
        a = Tensor(1, requires_grad=True)
        b = Tensor(2, requires_grad=True)
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
        a = Tensor(1, requires_grad=True)
        b = Tensor(2, requires_grad=True)
        c = Tensor(3, requires_grad=True)
        
        d = a + b
        e = d + c
        
        # e's parents should be d and c
        self.assertEqual(len(e.grad_fn.parents), 2)
        self.assertIn(d, e.grad_fn.parents)
        self.assertIn(c, e.grad_fn.parents)

    def test_int_plus_tensor(self):
        """Test: int + Tensor (uses __radd__)"""
        a = 5
        b = Tensor(10, requires_grad=True)
        c = a + b
        
        self.assertEqual(c.data, 15)
        self.assertEqual(c.grad_fn.operation, Operation.ADD)
        # Check that the first parent is the wrapped integer
        self.assertEqual(c.grad_fn.parents[0].data, 5)

    def test_graph_integrity(self):
        """Ensure the graph structure holds for scalar ops"""
        a = Tensor(1, requires_grad=True)
        b = 2
        c = a + b
        
        # This checks that we don't get the AttributeError
        for parent in c.grad_fn.parents:
            self.assertTrue(hasattr(parent, 'data'))
            self.assertTrue(hasattr(parent, 'grad_fn'))

    # --- Multiplication Tests ---
    def test_multiplication_data(self):
        """Simple multiplication: 10 * 20 = 200"""
        a = Tensor(10, requires_grad=True)
        b = Tensor(20, requires_grad=True)
        c = a * b
        self.assertEqual(c.data, 200)
        self.assertEqual(c.grad_fn.operation, Operation.MUL)

    def test_tensor_mul_int(self):
        """Test: Tensor * int (uses __mul__)"""
        a = Tensor(10, requires_grad=True)
        b = 3
        c = a * b
        self.assertEqual(c.data, 30)
        self.assertEqual(c.grad_fn.parents[0].data, 10)
        self.assertEqual(c.grad_fn.parents[1].data, 3)

    def test_int_mul_tensor(self):
        """Test: int * Tensor (uses __rmul__)"""
        a = 4
        b = Tensor(10, requires_grad=True)
        c = a * b
        self.assertEqual(c.data, 40)
        # Parent[0] should be the wrapped integer 4
        self.assertEqual(c.grad_fn.parents[0].data, 4)
        self.assertEqual(c.grad_fn.parents[1].data, 10)

    def test_mixed_ops_graph(self):
        """Test: (a * b) + c to ensure the graph handles mixed ops"""
        a = Tensor(2, requires_grad=True)
        b = Tensor(3, requires_grad=True)
        c = Tensor(4, requires_grad=True)
        
        d = a * b
        e = d + c
        
        self.assertEqual(e.data, 10)
        self.assertEqual(e.grad_fn.operation, Operation.ADD)
        self.assertEqual(e.grad_fn.parents[0].grad_fn.operation, Operation.MUL)

    # --- Subtraction Tests ---
    def test_subtraction_data(self):
        """Test: a - b"""
        a = Tensor(10, requires_grad=True)
        b = Tensor(3, requires_grad=True)
        c = a - b
        self.assertEqual(c.data, 7)
        self.assertEqual(c.grad_fn.operation, Operation.SUB)

    def test_int_minus_tensor(self):
        """Test: int - Tensor (5 - 10 = -5)"""
        a = 5
        b = Tensor(10, requires_grad=True)
        c = a - b
        self.assertEqual(c.data, -5)
        # Parent[0] should be the wrapped 5
        self.assertEqual(c.grad_fn.parents[0].data, 5)
        self.assertEqual(c.grad_fn.parents[1].data, 10)

    # --- ReLU Tests ---
    def test_relu_positive(self):
        """ReLU of positive should stay positive"""
        a = Tensor(5.0, requires_grad=True)
        b = a.relu()
        self.assertEqual(b.data, 5.0)
        self.assertEqual(b.grad_fn.operation, Operation.RELU)

    def test_relu_negative(self):
        """ReLU of negative should be zero"""
        a = Tensor(-5.0, requires_grad=True)
        b = a.relu()
        self.assertEqual(b.data, 0)
        self.assertEqual(b.grad_fn.parents[0], a)

# --- Topological Sort Tests ---
    def test_topological_sort_diamond(self):
        """
        Test a diamond graph:
           a
          / \
         b   c
          \ /
           d
        Ensures a is processed last and only once.
        """
        a = Tensor(2.0, requires_grad=True)
        b = a * 2.0
        c = a + 3.0
        d = b * c
        
        topo = d._build_toplogical_ordering()
        
        # 1. Check uniqueness
        self.assertEqual(len(topo), 4, "Should have exactly 4 nodes in the list")
        self.assertEqual(len(set(topo)), 4, "All nodes in topo sort should be unique")
        
        # 2. Check Order (Reversed Post-Order)
        # Root must be first, Leaf must be last
        self.assertEqual(topo[0], d)
        self.assertEqual(topo[-1], a)
        
        # 3. Check dependencies
        # b and c must appear after d but before a
        idx_a = topo.index(a)
        idx_b = topo.index(b)
        idx_c = topo.index(c)
        idx_d = topo.index(d)
        
        self.assertTrue(idx_d < idx_b < idx_a)
        self.assertTrue(idx_d < idx_c < idx_a)

    def test_topological_sort_diamond(self):
        """
        Test a diamond graph:
           a
          / \
         b   c
          \ /
           d
        """
        a = Tensor(2.0, requires_grad=True)
        b = a * 2.0  # Creates node b AND a constant Tensor(2.0)
        c = a + 3.0  # Creates node c AND a constant Tensor(3.0)
        d = b * c    # Creates node d
        
        topo = d._build_toplogical_ordering()
        
        # Change 4 to 6
        self.assertEqual(len(topo), 6, "Should have 6 nodes (a, b, c, d + 2 constants)")
        
        # The rest of your logic remains perfect
        self.assertEqual(topo[0], d)
        self.assertEqual(topo[-1], a)
        
        self.assertTrue(topo.index(d) < topo.index(b))
        self.assertTrue(topo.index(d) < topo.index(c))

    # --- Zero Grad Tests ---
    def test_zero_grad_logic(self):
        """Manually set grads and check if zero_grad clears the whole graph"""
        a = Tensor(2.0, requires_grad=True)
        b = Tensor(3.0, requires_grad=True)
        c = a * b
        
        # Manually pollute gradients
        a.grad = 1.0
        b.grad = 2.0
        c.grad = 3.0
        
        c.zero_grad()
        
        self.assertEqual(a.grad, 0.0)
        self.assertEqual(b.grad, 0.0)
        self.assertEqual(c.grad, 0.0)

    def test_zero_grad_no_leakage(self):
        """Check that requires_grad=False nodes don't get initialized with 0.0 grad"""
        a = Tensor(2.0, requires_grad=False)
        b = Tensor(3.0, requires_grad=True)
        c = a + b
        
        c.zero_grad()
        
        self.assertIsNone(a.grad, "Non-grad nodes should stay None")
        self.assertEqual(b.grad, 0.0, "Grad nodes should be zeroed")

if __name__ == "__main__":
    unittest.main()