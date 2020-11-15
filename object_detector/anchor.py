from bbox import BBox

class Anchor:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __rmul__(self, factor):
        out = Anchor(self.x * factor, self.y * factor, self.w * factor, self.h * factor)

        return out

    def __mul__(self, factor):
        out = Anchor(self.x * factor, self.y * factor, self.w * factor, self.h * factor)

        return out

    def to_box(self) -> BBox:
        x0 = (self.x - self.w * 0.5)
        x1 = (self.x + self.w * 0.5)
        y0 = (self.y - self.h * 0.5)
        y1 = (self.y + self.h * 0.5)

        box = BBox(x0, y0, x1, y1)

        return box

