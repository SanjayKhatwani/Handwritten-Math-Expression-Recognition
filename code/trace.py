import math

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

class trace:
    """
    Class for Trace information.
    """
    __slots__ = ('id', 'x', 'y', 'centroid_x', 'centroid_y', 'maxx', 'maxy',
                 'minx', 'miny', 'tgid', 'bb_center_x', 'bb_center_y')

    def __init__(self):
        self.id = 0
        self.x = []
        self.y = []
        self.centroid_x = 0
        self.centroid_y = 0
        self.tgid = -1

    def calculate_centroids(self):
        self.centroid_x = round(sum(self.x) / len(self.x), 2)
        self.centroid_y = round(sum(self.y) / len(self.y), 2)
        self.maxx = max(self.x)
        self.maxy = max(self.y)
        self.minx = min(self.x)
        self.miny = min(self.y)
        self.bb_center_x = self.minx + ((self.maxx - self.minx) / 2)
        self.bb_center_y = self.miny + ((self.maxy - self.miny) / 2)

    def get_distance_centroid(self, t2):
        return math.sqrt((self.centroid_x - t2.centroid_x) ** 2 +
                         (self.centroid_y - t2.centroid_y) ** 2)

    def get_distance_bb_center(self, t2):
        return math.sqrt((self.bb_center_x - t2.bb_center_x) ** 2 +
                         (self.bb_center_y - t2.bb_center_y) ** 2)

    def __str__(self):
        return str(self.id)