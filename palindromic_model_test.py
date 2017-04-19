'''Palindromic Model Test'''

import numpy as np
import tensorflow as tf
# palindromic methods
from palindromic_model import binary_encode
from palindromic_model import palindromic_encode

class PalindromicEncodeTest(tf.test.TestCase):
  def testBinaryEncode(self):
    with self.test_session():
      self.assertEqual(binary_encode(101,2**10).all(),np.array([101 >> d & 1 for d in range(2**10)]).all())

  def testNonPalindromicEncode(self):
    with self.test_session():
      self.assertEqual(palindromic_encode(10).all(),np.array([0,1,0]).all())
 
if __name__ == '__main__':
  tf.test.main()
