import mlx.core as mx
import torch

import unittest

from proteus.utils.utils import coerce_mx_to_torch, coerce_torch_to_mx


def mutate_slice(x: mx.array) -> mx.array:
    """Mutates a slice of the input array in-place"""
    # Create a slice of ones to copy in
    ones = mx.ones((1, x.shape[1]))
    # Update first row to be all ones
    x[0] = ones
    return x


class TestMLXMutation(unittest.TestCase):
    def test_mutate_slice(self):
        # Create test input
        x = mx.zeros((4, 4))
        torch_x = coerce_mx_to_torch(x)
        # Mutate the array
        print(x, torch_x)
        result = mutate_slice(x)
        print(x, torch_x)
        # Check first row is all ones
        self.assertTrue(mx.array_equal(result[0], mx.ones(4)))
        # Check rest of array is unchanged
        self.assertTrue(mx.array_equal(result[1:], mx.zeros((3, 4))))


if __name__ == "__main__":
    unittest.main()
