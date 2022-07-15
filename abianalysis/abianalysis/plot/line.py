from attrs import define, field

from abianalysis.plot.point import Point


def convert_point(point):
    if isinstance(point, Point):
        return point
    elif isinstance(point, tuple):
        if len(point) == 2:
            return Point(*point)
    else:
        raise ValueError("Unsupported type for line")


@define
class Line:
    fr: Point = field(converter=convert_point)
    to: Point = field(converter=convert_point)

    def to_tuple(self):
        return self.fr.to_tuple(), self.to.to_tuple()

    @classmethod
    def from_orth_points(cls, fr: Point, to: Point) -> 'Line':
        if fr.x == to.x:
            ymid = (fr.y + to.y) / 2
            return Line((ymid, fr.x - .5), (ymid, to.x + .5))
        if fr.y == to.y:
            xmid = (fr.x + to.x) / 2
            return Line((fr.y - .5, xmid), (to.y + .5, xmid))
