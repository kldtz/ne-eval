import unittest

from eval.centroids import *


class TestCentroids(unittest.TestCase):
    """
    Tests centroid evaluation.
    """

    def test_single_peak(self):
        xs = [(4, 7), (4, 11), (0, 7), (3, 7)]
        ys = [(0, 3), (8, 16), (11, 16), (7, 16)]
        gold = list(self.create_anns('x', xs)) + list(self.create_anns('y', ys))

        e = CentroidEvaluation(gold)

        centroids = list(e.type_counts['x'].centroids())
        self.assertEqual([(0, 1), (3, 1), (4, 2)], centroids[0].left)
        self.assertEqual([(6, 3), (10, 1)], centroids[0].right)
        centroids = list(e.type_counts['y'].centroids())
        self.assertEqual([(0, 1)], centroids[0].left)
        self.assertEqual([(2, 1)], centroids[0].right)
        self.assertEqual([(7, 1), (8, 1), (11, 1)], centroids[1].left)
        self.assertEqual([(15, 3)], centroids[1].right)

    def test_multiple_peaks(self):
        gold = [(4, 16), (8, 11), (12, 16)]
        self.assertRaises(Exception, CentroidEvaluation, self.create_anns('x', gold))

    def test_boundaries(self):
        gold = [(2, 6), (4, 6), (4, 8), (4, 10), (12, 18), (14, 17), (12, 19), (20, 23), (19, 22), (19, 23)]

        e = CentroidEvaluation(self.create_anns('x', gold))

        predicted = [(2, 8), (13, 17), (19, 23)]
        tp, fp, fn = e.evaluate(self.create_anns('x', predicted))
        self.assertEqual(3, len(tp))
        self.assertEqual(0, len(fp))
        self.assertEqual(0, len(fn))
        tp, fp, fn = e.evaluate(self.create_anns('x', predicted), lb=1, rb=1)
        self.assertEqual(2, len(tp))
        self.assertEqual(1, len(fp))
        self.assertEqual(1, len(fn))
        tp, fp, fn = e.evaluate(self.create_anns('x', predicted), lb=2, rb=1)
        self.assertEqual(1, len(tp))
        self.assertEqual(2, len(fp))
        self.assertEqual(2, len(fn))

    def test_print_smoke(self):
        gold = [(2, 6), (4, 6), (4, 8), (4, 10), (12, 18), (14, 17), (12, 19), (20, 23), (19, 22), (19, 23)]

        e = CentroidEvaluation(self.create_anns('x', gold))

        predicted = [(2, 8), (13, 17), (19, 23)]
        print(e.evaluate(self.create_anns('x', predicted), lb=2, rb=1))

    @staticmethod
    def create_anns(type, offsets):
        for start, end in offsets:
            yield (Annotation(type, start, end))


if __name__ == '__main__':
    unittest.main()
