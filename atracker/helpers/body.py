"""Silhouette measurements shared by background and colour detection."""
import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


BODY_MODES = ('centroid', 'skeleton')
HEAD_RULES = ('near_widest', 'near_centroid', 'far_centroid', 'unassigned')


def body_settings(body_mode='centroid', head_rule='near_widest', skeleton_points=10):
    if body_mode not in BODY_MODES:
        raise ValueError(f'body_mode must be one of {BODY_MODES}')
    if head_rule not in HEAD_RULES:
        raise ValueError(f'head_rule must be one of {HEAD_RULES}')
    if not isinstance(skeleton_points, int) or isinstance(skeleton_points, bool) or skeleton_points < 2:
        raise ValueError('skeleton_points must be an integer >= 2')
    return body_mode, head_rule, skeleton_points


def resample_line(points, count):
    """Return equally spaced points along arc length, including both tips."""
    points = np.asarray(points, dtype=float)
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] <= 0:
        raise ValueError('A centreline must have positive length')
    target = np.linspace(0, lengths[-1], count)
    return np.column_stack([np.interp(target, lengths, points[:, axis]) for axis in (0, 1)])


def ordered_skeleton(binary):
    """Trace the main connected skeleton path; discard branches and tiny islands.

    Two weighted graph sweeps recover the diameter for an unbranched/tree path.
    Closed loops have no biological tips and are left unresolved.
    """
    skeleton = cv2.ximgproc.thinning(binary)
    ys, xs = np.where(skeleton > 0)
    if len(xs) < 2:
        return None
    nodes = {(int(y), int(x)): i for i, (y, x) in enumerate(zip(ys, xs))}
    rows, cols, weights = [], [], []
    for (y, x), i in nodes.items():
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            j = nodes.get((y + dy, x + dx))
            if j is not None:
                # Avoid diagonal shortcuts across a right-angle bend.
                if dx and dy and ((y, x + dx) in nodes or (y + dy, x) in nodes):
                    continue
                rows.extend((i, j)); cols.extend((j, i))
                weights.extend((np.hypot(dx, dy),) * 2)
    graph = csr_matrix((weights, (rows, cols)), shape=(len(xs), len(xs)))
    _, labels = connected_components(graph)
    largest = np.argmax(np.bincount(labels))
    members = np.flatnonzero(labels == largest)
    degree = np.diff(graph.indptr)
    tips = members[degree[members] == 1]
    if len(tips) < 2:
        return None
    distance = dijkstra(graph, indices=int(tips[0]))
    start = int(tips[np.argmax(distance[tips])])
    distance, predecessor = dijkstra(graph, indices=start, return_predecessors=True)
    end = int(tips[np.argmax(distance[tips])])
    path = [end]
    while path[-1] != start:
        path.append(int(predecessor[path[-1]]))
    return np.column_stack((xs[path], ys[path])).astype(float)


def measure_body(contour, centroid, body_mode, head_rule, skeleton_points):
    """Measure in ROI-local pixels; head assignment is a configurable heuristic."""
    rect = cv2.minAreaRect(contour)
    major = float(max(rect[1]))
    result = dict(com=centroid, length=major, length_method='box_major_axis',
                  center_source='blob_centroid', body_valid=True)
    if body_mode == 'centroid':
        return result
    x, y, width, height = cv2.boundingRect(contour)
    origin = np.array((x - 2, y - 2))
    binary = np.zeros((height + 4, width + 4), np.uint8)
    cv2.drawContours(binary, [contour - origin], -1, 255, -1)
    path = ordered_skeleton(binary)
    result.update(body_valid=False, length=np.nan, length_method='centerline',
                  center_source='blob_centroid_fallback', head_assignment='unresolved')
    if path is None:
        return result
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, _, _, widest = cv2.minMaxLoc(distance)
    path += origin
    reference = np.asarray(widest) + origin if head_rule == 'near_widest' else np.asarray(centroid)
    distances = np.linalg.norm(path[[0, -1]] - reference, axis=1)
    head_first = distances[0] <= distances[1]
    if head_rule == 'far_centroid':
        head_first = not head_first
    if not head_first:
        path = path[::-1]
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    sampled = resample_line(path, skeleton_points)
    middle = resample_line(path, 3)[1]
    head, tail = tuple(path[0]), tuple(path[-1])
    result.update(com=tuple(middle), head=head, tail=tail, skeleton=sampled.tolist(),
                  length=length, body_straightness=float(np.linalg.norm(path[-1] - path[0]) / length),
                  center_source='centerline_midpoint', body_valid=True,
                  head_assignment=head_rule,
                  head_asymmetry=float(abs(distances[0]-distances[1]) / max(distances.max(), 1e-9)),
                  objbox=((x, y), (x + width, y + height)))
    if head_rule == 'unassigned':
        result['head_assignment'] = 'unassigned'
    return result
