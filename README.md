
# Palindromic model

[![Build Status](https://travis-ci.org/kaiquewdev/palindromic_model.svg?branch=master)](https://travis-ci.org/kaiquewdev/palindromic_model)

# Description

This a basic model to describe a palindromic model using binary encode based on this article

[Fizz Buzz using Tensorflow as environment](http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/)

# Primary aspects of the model

First of all this code are based on import numpy and tensorflow as an aliased:

```
import numpy as np
import tensorflow as tf
```

Wheve gonna make a model based on multi-layer-perceptron with one hidden layer or a neural network.
To do that whe need to compose a vector of "activations" with the following above method:

```
binary_encode = lambda i,num_digits: np.array([i >> d & 1 for d in range(num_digits)])
```

Outputting those information need to determine what is a palindromic where first position indicates
"print as-is", the second indicates a "palindromic", and then:

```
def palindromic_encode(i):
    if str(i)[::-1] == str(i): return np.array([0,0,1])
    elif str(i)[::-1] == str(i): return np.array([0,1,0])
    else: return np.array([1,0,0])
```

Training data could be use a sequence generation of 1 to 100 on the set, in total the numbers to be trainned was 1024.

```
NUM_DIGITS = 10
trX = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2**NUM_DIGITS)])
trY = np.array([palindromic_encode(i) for i in range(101, 2**NUM_DIGITS)])
```

Now we setup the model using tensorflow on that case another tool set can be used like theano.
Using at least 100 hidden units.

```
NUM_HIDDEN = 100
```