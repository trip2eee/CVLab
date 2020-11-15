class BBox:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def __rmul__(self, factor):
        out = BBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

        return out

    def __mul__(self, factor):
        out = BBox(self.x0 * factor, self.y0 * factor, self.x1 * factor, self.y1 * factor)

        return out

    def compute_iou(self, box1):

        return 0.0

