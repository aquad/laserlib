#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Test numpy to eigen conversion, will throw under error.
"""

import numpy as np
import LaserPy

foo = np.zeros([100,3,3], dtype=np.float32)
LaserPy._test_numpy_float_to_eigen(foo, 'foo', 100, 9)


