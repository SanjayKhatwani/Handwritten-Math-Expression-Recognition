__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

class object:

    __slots__ = ('traceids', 'id', 'label', 'bbx', 'bby', 'bbminx', 'bbmaxx',
                 'bbminy', 'bbmaxy')

    def __init__(self, idd, l, trids=[]):
        self.traceids = trids
        self.id = idd
        self.label = l
        self.bbx = 0
        self.bby = 0
        self.bbminx = 0
        self.bbmaxx = 0
        self.bbminy = 0
        self.bbmaxy = 0