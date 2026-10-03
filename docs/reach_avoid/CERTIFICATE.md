# RA-SEP-1 sufficient certificate (prospective)

Implementation: `experiments/reach_avoid/certificate.py`; executed-geometry and
substep audit: `experiments/reach_avoid/mechanisms.py`; constructors:
`experiments/reach_avoid/fixtures.py`; frozen pilot: `mechanism_protocol.json`.

For each actual target collision box, transform the target body root into that
box's local coordinates. If half sizes are s_i and root coordinates u_i, the
ball of radius r=min_i(s_i-|u_i|)>0 around the root is inside that box. Use the
largest independently verified such radius, recording the executing box ID,
actual sizes and transformed root. This is an inner bound, not a visual mesh
bounding sphere. Every rigid orientation of the target still contains that ball.
Consequently a WORLD-aligned cube with halfwidth rho=r/sqrt(3)-1 micrometre is
contained in the target under every orientation. The negative proof never fixes
the orientation of the milk or the gripper.

Expand each actual static axis-aligned red box by rho in all three directions.
This is the exact center obstacle for the contained cube. If the milk root is
in that expanded box, a subset of milk material intersects the red box. The
explicit digital safety contract forbids each adjacent synchronized-integration
root segment from intersecting this expanded box. It also forbids native and
synchronized whole-robot/target protected contacts and the earlier actor-center
red-box crossing guard. The swept-root rule strengthens sampled native contact
safety: soft penetration or integration-step tunneling is not an escape. It is
NOT a claim that MuJoCo contact alone implements this rule. A freely moving
irrelevant object or basket does not move the world-fixed red obstacles.

Choose a closed rectangular region with the initial root strictly inside and
the fixed initial-world goal wholly disjoint. Intersect the ACTUAL expanded
obstacles with each of its six faces. A separate checker partitions each face
at every rectangle boundary, including boundary coordinates, and verifies full
y-interval union throughout every x interval. Thus faces, face edges and corners
are covered without a sampled-grid shortcut or equality to a constructor. A
continuous piecewise-linear root path to the fixed goal must leave this region
and intersect a covered boundary. That violates the explicit guard. This excludes
all action trajectories, object pushes, grasp orientations, routes over/under/
around the region and any allowed contact mode under RA-SEP-1. The original goal
AND synchronized goal AND fixed-world goal is required, so moving the basket
into the region does not create a loophole. No speed bound is assumed.

The expected milk radius is approximately25.88mm, implying a sufficient uniform
slit-width bound near29.88mm for this cube certificate. These numbers are only
expectations until read from each executing model. The cube bound is conservative;
it is NOT the true feasibility transition. Widths .034/.060/.180m without a
certificate require a complete safe witness or remain UNKNOWN. Report the gap
between largest certified width and smallest witnessed width, plus interior
unknown case counts, without pretending the entire gap is certified negative.

All variants of each layout restore one naturally settled, unobstructed canonical
state. Disposable fixture-construction settling is archived and overwritten only
by that canonical state before scored actions; all compared initial arrays are
saved. Initial overlap fails validity. Removing the roof must fail the separator
coverage test. The irrelevant fixture cannot certify a root outside its region.
Safe witnesses use the original bounded OSC/gripper action interface, T=300
execution commands, and the same complete safety/goal checks as negatives.

Slit and bar cage are different visible constructions of this same separator
proof family. They do not establish two independent dynamical mechanisms.
