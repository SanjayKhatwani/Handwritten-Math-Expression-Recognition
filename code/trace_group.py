__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

class trace_group:
    """
    Class for trace groups.
    """
    __slots__ = ('id', 'trace_list', 'annotation_mml', 'mml_href',
                 'bbx', 'bby', 'bbminx', 'bbmaxx', 'bbminy', 'bbmaxy')

    def __init__(self):
        self.id = 0
        self.trace_list = []
        self.annotation_mml = ''
        self.mml_href = ''
        self.bbx = 0
        self.bby = 0
        self.bbminx = 0
        self.bbmaxx = 0
        self.bbminy = 0
        self.bbmaxy = 0
