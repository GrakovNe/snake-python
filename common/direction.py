from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def dx(self):
        return self.value[0]

    @property
    def dy(self):
        return self.value[1]

    def opposite(self, other: "Direction") -> bool:
        return self.dx == -other.dx and self.dy == -other.dy