import xml.etree.ElementTree as ET

from graph import Graph
from trace import trace
from trace_group import trace_group
from object import object
from relation import relation

__author__ = ['Dharmendra Hingu', 'Sanjay Khatwani']


class inkml:
    """
    Main inkml object
    """
    __slots__ = ('ground_truth', 'math_ml', 'traces', 'trace_groups',
                 'objects', 'relations')

    def __init__(self):
        self.ground_truth = ''
        self.math_ml = ''
        self.traces = []
        self.trace_groups = []
        self.objects = []
        self.relations = []

    def get_trace(self, identity):
        for trace in self.traces:
            if trace.id == identity:
                return trace
        return None

    def get_trace_group(self, identity):
        for tg in self.trace_groups:
            if tg.id == identity:
                return tg
        return None

    def get_object(self, identity):
        for obj in self.objects:
            if obj.id == identity:
                return obj
        return None

    def get_traces_in_group(self, identity):
        """
        Get traces in a group.
        """
        traces_ids_in_group = []
        traces_in_group = []
        for tg in self.trace_groups:
            if tg.id == identity:
                traces_ids_in_group = tg.trace_list
        for trace in self.traces:
            if trace.id in traces_ids_in_group:
                traces_in_group.append(trace)
        return traces_in_group

    def get_traces_in_object(self, identity):
        """
        Get traces in a group.
        """
        traces_ids_in_object = []
        traces_in_object = []
        for tg in self.objects:
            if tg.id == identity:
                traces_ids_in_object = tg.traceids
        for trace in self.traces:
            if trace.id in traces_ids_in_object:
                traces_in_object.append(trace)
        return traces_in_object

    def get_relation_between(self, o1, o2):
        for r in self.relations:
            if o1 == r.object_ids[0] and o2 == r.object_ids[1]:
                return [r.label, r.id]
        return [None, None]

    def compute_all_tg_bb(self):
        for tg in self.trace_groups:
            trs = self.get_traces_in_group(tg.id)
            tg.bbminx = min([x.minx for x in trs])
            tg.bbmaxx = max([x.maxx for x in trs])
            tg.bbminy = min([x.miny for x in trs])
            tg.bbmaxy = max([x.maxy for x in trs])
            tg.bbx = tg.bbminx + (tg.bbmaxx - tg.bbminx) / 2
            tg.bby = tg.bbminy + (tg.bbmaxy - tg.bbminy) / 2

    def compute_all_obj_bb(self):
        for tg in self.objects:
            trs = self.get_traces_in_object(tg.id)
            tg.bbminx = min([x.minx for x in trs])
            tg.bbmaxx = max([x.maxx for x in trs])
            tg.bbminy = min([x.miny for x in trs])
            tg.bbmaxy = max([x.maxy for x in trs])
            tg.bbx = tg.bbminx + (tg.bbmaxx - tg.bbminx) / 2
            tg.bby = tg.bbminy + (tg.bbmaxy - tg.bbminy) / 2

    def build_association_graph(self):
        """
        Build complete association graph for the inkml object.
        """
        g = Graph(len(self.traces))
        for tr in self.traces:
            g.add_node(tr.id)

        for tr_id_i in range(len(self.traces)):
            for tr_id_j in range(tr_id_i + 1, len(self.traces)):
                dist = self.traces[tr_id_i].get_distance_centroid(self.traces[tr_id_j])
                g.add_edge(self.traces[tr_id_i].id, self.traces[tr_id_j].id,
                           dist)
        return g


def marshal_inkml(file):
    """
    Build inkml object from a inkml file.
    """
    this_inkml = inkml()
    tree = ET.parse(file)
    root = tree.getroot()
    this_inkml.ground_truth = get_ground_truth(root)
    this_inkml.traces = get_traces(root)
    this_inkml.trace_groups = get_trace_groups(root)
    assign_trace_group_ids_to_traces(this_inkml.trace_groups, this_inkml)
    return this_inkml


def get_ground_truth(root):
    """
    Extract GT from a inkml file.
    """
    for annotation in root.findall('{http://www.w3.org/2003/InkML}annotation'):
        if annotation.get('type') == 'truth':
            return annotation.text


def get_traces(root):
    """
    Extract traces from a inkml file.
    """
    trace_objs = []
    for trac in root.findall('{http://www.w3.org/2003/InkML}trace'):
        this_trace = trace()
        this_trace.id = trac.get('id')
        data = trac.text.rstrip().lstrip()
        xy = data.split(',')
        for one in xy:
            one_xy = one.rstrip().lstrip().split(' ')
            this_trace.x.append(float(one_xy[0]))
            this_trace.y.append(float(one_xy[1]))
        this_trace.calculate_centroids()
        trace_objs.append(this_trace)
    return trace_objs


def get_trace_groups(root):
    """
    Extract trace groups from a inkml file.
    """
    trace_group_objs = []
    main_tg = root.find('{http://www.w3.org/2003/InkML}traceGroup')
    if main_tg is not None:
        for tg in main_tg.findall('{http://www.w3.org/2003/InkML}traceGroup'):
            this_tg = trace_group()
            this_tg.id = tg.get('{http://www.w3.org/XML/1998/namespace}id')
            for annotation in tg.findall(
                    '{http://www.w3.org/2003/InkML}annotation'):
                if annotation.get('type') == 'truth':
                    this_tg.annotation_mml = annotation.text
            this_tg.mml_href = tg.find('{'
                                       'http://www.w3.org/2003/InkML}annotationXML')
            if this_tg.mml_href is not None:
                this_tg.mml_href = this_tg.mml_href.get('href')
            for trace in tg.findall('{http://www.w3.org/2003/InkML}traceView'):
                this_tg.trace_list.append(trace.get('traceDataRef'))
            trace_group_objs.append(this_tg)
    return trace_group_objs

def assign_trace_group_ids_to_traces(tgs, inkmlobj):
    for tg in tgs:
        for tr in tg.trace_list:
            inkmlobj.get_trace(tr).tgid = tg.id

def marshal_objects_relations(file, inkml_obj):
    with open(file, 'r') as file_obj:
        lines = list(file_obj)
    objs = []
    rels = []
    for line in lines:
        if line.startswith('O'):
            objs.append(line.strip().split(','))
        elif line.startswith('R'):
            rels.append(line.strip().split(','))

    for obj in objs:
        identity = obj[1].strip()
        label = obj[2]
        trs = [i.strip() for i in obj[4:]]
        new_obj = object(identity, label, trs)
        inkml_obj.objects.append(new_obj)

    for rel in rels:
        objids = [i.strip() for i in rel[1:3]]
        lab = rel[3].strip()
        weight = float(rel[4].strip())
        iden = objids[0]+''+objids[1]
        new_rel = relation(idd = iden, objids=objids, lab=lab, w=weight)
        inkml_obj.relations.append(new_rel)

    inkml_obj.compute_all_obj_bb()

