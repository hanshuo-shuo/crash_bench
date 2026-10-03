"""Independent sufficient separator certificate, using only executed geometry.

This proves impossibility under an explicitly stated digital swept-center guard.
It does not infer continuous rigid contact from MuJoCo's soft contact solver.
"""
import math


def inscribed_radius(box_half_sizes, root_in_box_frame):
    """Largest root-centered ball contained in ONE actual collision box."""
    if len(box_half_sizes) != 3 or len(root_in_box_frame) != 3:
        raise ValueError('Expected 3D box')
    if not all(math.isfinite(x) for x in box_half_sizes + root_in_box_frame):
        raise ValueError('Nonfinite material geometry')
    return max(0., min(s-abs(x) for s, x in zip(box_half_sizes, root_in_box_frame)))


def rectangle_covered(bounds, rectangles):
    """Exact finite partition proof of closed axis-aligned rectangle coverage.

    Interior membership changes only at rectangle x boundaries. Test each open
    partition interval and the boundaries, with complete y-interval union. No
    sampled grid or construction-name assumption is used.
    """
    x0, y0, x1, y1 = bounds
    if x0 >= x1 or y0 >= y1: raise ValueError('Degenerate face')
    xs = sorted({x0, x1} | {v for r in rectangles for v in (r[0], r[2]) if x0 < v < x1})
    # Test a rectangle's coverage of the entire partition interval. Computing a
    # floating midpoint can round to an endpoint and hide a one-ULP open gap.
    checks = [(x,x) for x in xs] + list(zip(xs,xs[1:]))
    for left, right in checks:
        intervals = sorted((max(y0, r[1]), min(y1, r[3])) for r in rectangles
            if r[0] <= left and r[2] >= right and r[3] >= y0 and r[1] <= y1)
        end = y0
        for lo, hi in intervals:
            if lo > end: return False
            end = max(end, hi)
        if end < y1: return False
    return True


def certify(boxes, lower, upper, initial_root, fixed_goal, radius, initial_safe,
            all_static, material_verified, guard_active, numerical_margin=1e-6):
    """A fixed cube of half-width r/sqrt(3) fits inside the orientation-free ball.

    Expand actual obstacles by this cube. If they cover every face of an initial
    containing box, no admissible center polyline reaches a disjoint fixed goal.
    The all-route theorem is independent of controller, pushing, rotations, T,
    and the number of solver iterations. T only constrains feasible witnesses.
    """
    values = lower+upper+initial_root+fixed_goal[0]+fixed_goal[1]+[radius, numerical_margin]
    if not all(math.isfinite(x) for x in values): raise ValueError('Nonfinite certificate input')
    rho = max(0., radius/math.sqrt(3)-numerical_margin)
    expanded = []
    for box in boxes:
        lo, hi = box['lower'], box['upper']
        if not all(math.isfinite(x) for x in lo+hi) or any(a >= b for a, b in zip(lo, hi)):
            raise ValueError('Invalid actual obstacle')
        expanded.append(([x-rho for x in lo], [x+rho for x in hi]))
    faces = []
    for axis in range(3):
        tangents = [i for i in range(3) if i != axis]
        a, b = tangents
        for coordinate in (lower[axis], upper[axis]):
            rectangles = [[lo[a], lo[b], hi[a], hi[b]] for lo, hi in expanded if lo[axis] <= coordinate <= hi[axis]]
            faces.append(rectangle_covered([lower[a], lower[b], upper[a], upper[b]], rectangles))
    checks = dict(all_six_faces_covered=all(faces), positive_material_ball=radius > 0 and rho > 0,
        material_geometry_verified=bool(material_verified), actual_obstacles_world_fixed=bool(all_static),
        initial_contact_safe=bool(initial_safe), digital_guard_active=bool(guard_active),
        initial_root_strictly_inside=all(a < x < b for a, x, b in zip(lower, initial_root, upper)),
        frozen_goal_disjoint=any(fixed_goal[1][i] < lower[i] or fixed_goal[0][i] > upper[i] for i in range(3)))
    return dict(label='infeasible' if all(checks.values()) else 'unknown',
        contract='RA-SEP-1', conditions=checks, covered_faces=faces,
        material_ball_radius=radius, world_cube_halfwidth=rho,
        sufficient_uniform_slit_width=2*rho,
        scope='Conditional swept-center digital contract, all target orientations and root routes; not unrestricted physical impossibility.',
        proof='A world-aligned cube inside an orientation-invariant material ball intersects an obstacle whenever its center enters that cube-expanded obstacle. All six faces of a box containing the initial root are covered by expanded actual obstacles. Every center polyline to the disjoint frozen goal crosses a covered face; the guard forbids it.')


def classify(witness_safe_success, witness_steps, horizon_T, certificate):
    witnessed = witness_safe_success and witness_steps <= horizon_T
    if witnessed and certificate['label'] == 'infeasible': raise RuntimeError('Contradictory independent evidence')
    return 'feasible' if witnessed else certificate['label']
