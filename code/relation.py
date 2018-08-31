__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']

class relation:

    __slots__ = ('object_ids', 'label', 'weight', 'id')

    def __init__(self, idd=0, objids=[], lab='', w=0):
        self.object_ids = objids
        self.label = lab
        self.weight = w
        self.id = idd