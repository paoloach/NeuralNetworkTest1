class PointGroup:
    def __init__(self,x,y,value, logistic):
        self.data = [(x,y,logistic)]
        self.value = value

    def add(self, x, y, logistic):
        self.data.append((x, y,logistic))

    def is_of_group(self, x, y, value):
        if self.value != value:
            return False
        for (gx, gy, _) in self.data:
            dist = (gx - x) *(gx-x) + (gy - y) * (gy-y)
            if dist <= 4:
                return True
        return False

    def max_point(self):
        max_point = (0,0,0)
        left=100000
        right=0
        top=100000
        bottom=0
        for (y,x,logistic) in self.data:
            if y < top:
                top=y
            if y > bottom:
                bottom = y
            if x < left:
                left = x
            if x > right:
                right = x
            if max_point[2] < logistic:
                max_point = (x,y,logistic)

        return (max_point[1]+8, max_point[0]+8, 8+(top+bottom)/2,8+(right+left)/2,max_point[2],self.value)

class ListGroup:
    def __init__(self):
        self.data = []

    def add(self, x, y, value, logistic):
        added = False
        for group in self.data:
            if group.is_of_group(x, y, value):
                group.add(x, y, logistic)
                added = True
                break
        if not added:
            self.data.append(PointGroup(x,y,value, logistic))

    def size(self):
        return len(self.data)

    def get(self, i):
        return self.data[i].max_point()
