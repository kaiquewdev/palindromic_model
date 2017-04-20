'''Palindromic Model Test'''

import numpy as np
import tensorflow as tf
# palindromic methods
from palindromic_model import binary_encode
from palindromic_model import palindromic_encode
from palindromic_model import palindromic

class BinaryEncodeTest(tf.test.TestCase):
  def testBinaryEncode(self):
    with self.test_session():
      self.assertEqual(binary_encode(101,2**10).all(),np.array([101 >> d & 1 for d in range(2**10)]).all())

class PalindromicEncodeTest(tf.test.TestCase):
  def testNonPalindromicEncode(self):
    with self.test_session():
      self.assertEqual(palindromic_encode(10).all(),np.array([0,1,0]).all())
 
class PalindromicStatesTest(tf.test.TestCase):
  def testFirstPureState(self):
    with self.test_session():
      self.assertEqual(palindromic(10,0),'10')

  def testFirstPalindromicState(self):
    with self.test_session():
      self.assertEqual(palindromic(101,1),'This numbers is a palindromic (101)')

  def testSecondPalindromicState(self):
    with self.test_session():
      self.assertEqual(palindromic(1001,2),'This numbers is a palindromic (1001)')

if __name__ == '__main__':
  tf.test.main()
