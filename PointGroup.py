class PointGroup:
    def __init__(self,x,y,value):
        self.data = [(x,y)]
        self.value = value

    def add(self, x, y, value):
        self.data.append((x, y))
        self.value = value

    def is_of_group(self, x, y, value):
        if self.value != value:
            return False
        for (gx, gy) in self.data:
            dist = (gx - x) *(gx-x) + (gy - y) * (gy-y)
            if dist <= 4:
                return True
        return False


class ListGroup:
    def __init__(self):
        self.data = []

    def add(self, x, y, value):
        added = False
        for group in self.data:
            if group.is_of_group(x, y, value):
                group.add(x, y, value)
                added = True
                break
        if not added:
            self.data.append(PointGroup(x,y,value))

    def size(self):
        return len(self.data)
