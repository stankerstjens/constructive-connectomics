from typing import Iterable

from attrs import define, field


@define(order=True, frozen=True)
class Point:
    x: int = field()
    y: int = field()

    def left(self) -> 'Point':
        return Point(self.x - 1, self.y)

    def up(self) -> 'Point':
        return Point(self.x, self.y + 1)

    def down(self) -> 'Point':
        return Point(self.x, self.y - 1)

    def right(self) -> 'Point':
        return Point(self.x + 1, self.y)

    def around(self) -> 'Iterable[Point]':
        yield self.left()
        yield self.right()
        yield self.up()
        yield self.down()

    def to_tuple(self):
        return self.x, self.y
