from collections import namedtuple, defaultdict, OrderedDict

Annotation = namedtuple('Annotation', ['type', 'start', 'end'])
EvalSets = namedtuple('EvalSets', ['tp', 'fp', 'fn'])


class _Centroid:
    __slots__ = ['left', 'right', 'type']

    def __init__(self):
        self.left = []
        self.right = []
        self.type = None

    @property
    def min(self):
        return self.left[-1][0], self.right[0][0]

    @property
    def max(self):
        return self.left[0][0], self.right[-1][0]

    @property
    def single_peak(self):
        return self.left[-1] < self.right[0]

    def __repr__(self):
        return "Centroid(type='{}', left=[{}], right=[{}])".format(self.type, ', '.join(
            str(x[0]) + ':' + str(x[1]) for x in self.left), ', '.join(
            str(x[0] + 1) + ':' + str(x[1]) for x in self.right))


class _Transition:
    __slots__ = ['votes', 'diff', 'centroid']

    def __init__(self):
        self.votes = 0
        self.diff = 0
        self.centroid = None


class _TypeCounts:
    def __init__(self):
        self.counts = OrderedDict()
        self._centroids = []

    def centroids(self, threshold=0):
        for c in self._centroids:
            if self.counts[c.left[-1][0]].votes >= threshold:
                yield c

    def validate(self):
        for centroid in self._centroids:
            if not centroid.single_peak:
                raise Exception('Centroid has more than one center: {}!'.format(centroid))
        return True

    def add_annotation(self, annotation):
        for i in range(annotation.start, annotation.end - 1):
            if i not in self.counts.keys():
                self.counts[i] = _Transition()
            self.counts[i].votes += 1
        if i + 1 not in self.counts.keys():
            self.counts[i + 1] = _Transition()

    def collect_centroids(self):
        centroid = None
        prev_val = 0
        for i in self.counts.keys():
            if prev_val == 0 and self.counts[i].votes == 0:
                centroid = None
            if self.counts[i].votes >= 1 and prev_val == 0:
                centroid = _Centroid()
                self._centroids.append(centroid)
                self.counts[i].diff = self.diff(i, prev_val)
                centroid.left.append((i, self.diff(i, prev_val)))
            elif self.counts[i].votes > prev_val:
                self.counts[i].diff = self.diff(i, prev_val)
                centroid.left.append((i, self.diff(i, prev_val)))
            elif self.counts[i].votes < prev_val:
                self.counts[i].diff = self.diff(i, prev_val)
                centroid.right.append((i, self.diff(i, prev_val)))
            if centroid is not None:
                self.counts[i].centroid = centroid
            prev_val = self.counts[i].votes

    def diff(self, i, prev_val):
        return abs(self.counts[i].votes - prev_val)

    def match_centroid(self, annotation, lb=0, rb=0, threshold=0):
        centroid = self.counts[annotation.start].centroid
        if (centroid is not None and
                centroid is self.counts[annotation.end - 1].centroid and
                self.counts[centroid.left[-1][0]].votes >= threshold and
                self.counts[annotation.start].diff >= lb and
                self.counts[annotation.end - 1].diff >= rb):
            centroid.type = annotation.type
            return centroid
        return None


class CentroidEvaluation:
    Centroid = namedtuple('Centroid', ['type', 'left', 'right'])

    def __init__(self, gold_annotations):
        self.type_counts = defaultdict(_TypeCounts)
        self._add_annotations(gold_annotations)
        self._compute_centroids()

    def _add_annotations(self, annotations):
        for annotation in sorted(annotations, key=lambda x: x.start):
            self.type_counts[annotation.type].add_annotation(annotation)

    def _compute_centroids(self):
        for counts in self.type_counts.values():
            counts.collect_centroids()
            counts.validate()

    def evaluate(self, predictions, threshold=0, lb=0, rb=0):
        tp, fp = [], []
        tp_centroids = set()
        for annotation in predictions:
            if annotation.type in self.type_counts:
                centroid = self.type_counts[annotation.type].match_centroid(
                    annotation, threshold=threshold, lb=lb, rb=rb)
                if centroid:
                    tp_centroids.add(centroid)
                    tp.append(annotation)
                else:
                    fp.append(annotation)
            else:
                fp.append(annotation)
        return EvalSets(tp, fp, self._collect_false_negatives(tp_centroids, threshold, lb, rb))

    def _collect_false_negatives(self, tp_centroids, threshold, lb, rb):
        fn = []
        for ann_type, counts in self.type_counts.items():
            for c in counts.centroids(threshold=threshold):
                if c not in tp_centroids:
                    left = [l[0] for l in c.left if l[1] >= lb]
                    right = [r[0] for r in c.right if r[1] >= rb]
                    fn.append(self.Centroid(ann_type, tuple(left), tuple(right)))
        return fn
