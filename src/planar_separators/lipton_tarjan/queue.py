import numpy as np


class Queue:

    def __init__(self):

        # Hacks to make numba work
        self._left_stack = list(np.zeros(0, dtype=np.int32))
        self._right_stack = list(np.zeros(0, dtype=np.int32))

    def append(self, value):

        self._right_stack.append(value)

    def popleft(self):

        if len(self._left_stack) == 0:

            self._right_stack.reverse()
            self._left_stack = self._right_stack
            self._right_stack = list(np.zeros(0, dtype=np.int32))

        return self._left_stack.pop()

    def is_empty(self):

        return len(self._left_stack) + len(self._right_stack) == 0
