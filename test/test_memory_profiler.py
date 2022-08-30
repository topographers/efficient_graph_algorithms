#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 01:52:34 2022

@author: hanlin
"""

from memory_profiler import profile


@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a

def test_memory_profiler():
    my_func()
