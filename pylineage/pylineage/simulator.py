#  MIT License
#
#  Copyright (c) 2022. Stan Kerstjens
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
from heapq import heappop, heappush
from typing import Callable


class Event:
    def __init__(self, time: float, action: Callable):
        self.time = time
        self.action = action

    def __call__(self, *args, **kwargs):
        return self.action(*args, **kwargs)

    def __lt__(self, other):
        return self.time.__lt__(other.time)

    def __le__(self, other):
        return self.time.__le__(other.time)

    def __gt__(self, other):
        return self.time.__gt__(other.time)

    def __ge__(self, other):
        return self.time.__ge__(other.time)


class Simulator:
    def __init__(self):
        self.time = 0
        self._queue = []

    @property
    def is_finished(self):
        return len(self._queue) == 0

    def run_until_time(self, finish_time):
        while self.time < finish_time:
            self.tick()

    def clear(self):
        self._queue = []

    def run(self):
        while not self.is_finished:
            self.tick()

    def run_while(self, condition: Callable):
        while condition():
            self.tick()

    def schedule(self, delay: float, action: Callable):
        heappush(self._queue, Event(self.time + delay, action))

    def tick(self):
        event = heappop(self._queue)
        self.time = event.time
        event()
