"""
Convert MediaPipe's canonical_face_model.obj into a complete head mesh and
emit it as a TypeScript module for the dashboard.

The canonical model is a FACE SHELL: anatomically real topology (the same
468-point mesh used by face-tracking systems) but open at the back. To make
it read as a head rather than a floating mask, the open boundary loop is
swept backwards over an ellipsoidal profile to form a cranium, then closed
at a pole. A short neck is extruded from the jaw underside.

Output: positions (Float32Array), triangle indices, and a deduplicated
wireframe edge list — everything the renderer needs, no runtime parsing.
"""
import sys
from collections import Counter, defaultdict

import numpy as np

SRC = sys.argv[1]
OUT = sys.argv[2]

# ── load ─────────────────────────────────────────────────────────────────────
verts, faces = [], []
for line in open(SRC):
    p = line.split()
    if not p:
        continue
    if p[0] == "v":
        verts.append([float(x) for x in p[1:4]])
    elif p[0] == "f":
        faces.append([int(x.split("/")[0]) - 1 for x in p[1:4]])

V = np.array(verts, dtype=np.float64)
F = np.array(faces, dtype=np.int32)
n_face_verts = len(V)

# ── boundary loop (edges used by exactly one triangle) ───────────────────────
edge_count = Counter()
for a, b, c in F:
    for e in ((a, b), (b, c), (c, a)):
        edge_count[tuple(sorted(e))] += 1
boundary = [e for e, n in edge_count.items() if n == 1]

adj = defaultdict(list)
for a, b in boundary:
    adj[a].append(b)
    adj[b].append(a)

start = boundary[0][0]
loop = [start]
prev, cur = None, start
while True:
    nxts = [x for x in adj[cur] if x != prev]
    if not nxts:
        break
    nxt = nxts[0]
    if nxt == start:
        break
    loop.append(nxt)
    prev, cur = cur, nxt
loop = np.array(loop, dtype=np.int32)
print(f"boundary loop: {len(loop)} verts")

# ── orient the loop counter-clockwise seen from behind (+? ) so generated
# cranium triangles keep the same winding as the face shell ──────────────────
C = V[loop].mean(axis=0)
ang = np.arctan2(V[loop][:, 1] - C[1], V[loop][:, 0] - C[0])
if np.sum(np.diff(np.unwrap(ang))) < 0:
    loop = loop[::-1]

# ── build the cranium: sweep the loop back over an ellipsoid profile ────────
loop_pts = V[loop]
loop_c = loop_pts.mean(axis=0)
# how far back the skull reaches, proportional to face depth
depth = V[:, 2].max() - V[:, 2].min()
back_z = loop_c[2] - depth * 1.05

RINGS = 14
new_verts = []
ring_idx = [loop]  # index arrays, ring 0 is the existing boundary

for i in range(1, RINGS + 1):
    t = i / RINGS                      # 0 -> at rim, 1 -> at back pole
    # Ellipsoidal sweep: width follows cos, depth follows sin, so the rings
    # trace a proper half-ellipsoid. Holding the width open early (sqrt)
    # keeps the skull round instead of tapering to a cone.
    scale = float(np.cos(t * np.pi / 2) ** 0.5)
    z = float(loop_c[2] + (back_z - loop_c[2]) * np.sin(t * np.pi / 2))
    ring = (loop_pts - loop_c) * scale + loop_c
    ring[:, 2] = z
    base = n_face_verts + len(new_verts)
    new_verts.extend(ring.tolist())
    ring_idx.append(np.arange(base, base + len(loop), dtype=np.int32))

# pole vertex closing the very back
pole = np.array([loop_c[0], loop_c[1], back_z])
pole_i = n_face_verts + len(new_verts)
new_verts.append(pole.tolist())

# ── neck: a short column under the jaw ──────────────────────────────────────
# Disabled: the design calls for a complete face with NO neck. The cranium
# bridging above already closes the entire boundary loop, so dropping the
# neck leaves a clean, watertight head that ends at the jaw — no open hole.
# Flip BUILD_NECK back to True to restore the short neck column.
BUILD_NECK = False
neck_rings = []
if BUILD_NECK:
    ys = loop_pts[:, 1]
    jaw_mask = ys < (ys.min() + (ys.max() - ys.min()) * 0.30)
    jaw_ids = loop[jaw_mask]
    NECK = 5
    neck_rings = [jaw_ids]
    if len(jaw_ids) >= 3:
        jaw_pts = V[jaw_ids]
        jaw_c = jaw_pts.mean(axis=0)
        for i in range(1, NECK + 1):
            t = i / NECK
            ring = (jaw_pts - jaw_c) * (1.0 - 0.12 * t) + jaw_c
            ring[:, 1] = jaw_pts[:, 1] - (depth * 0.52) * t
            # drift slightly back: a neck sits under the skull, not the chin
            ring[:, 2] = jaw_pts[:, 2] - depth * 0.28 * t
            base = n_face_verts + len(new_verts)
            new_verts.extend(ring.tolist())
            neck_rings.append(np.arange(base, base + len(jaw_ids), dtype=np.int32))

ALL = np.vstack([V, np.array(new_verts, dtype=np.float64)])

# ── stitch triangles ─────────────────────────────────────────────────────────
tris = [F]


def bridge(a, b, closed=True):
    """Triangle strip between two equal-length index rings."""
    out = []
    n = len(a)
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        j = (i + 1) % n
        out.append([a[i], b[i], b[j]])
        out.append([a[i], b[j], a[j]])
    return np.array(out, dtype=np.int32)


for i in range(len(ring_idx) - 1):
    tris.append(bridge(ring_idx[i], ring_idx[i + 1]))

last = ring_idx[-1]
tris.append(np.array([[last[i], pole_i, last[(i + 1) % len(last)]]
                      for i in range(len(last))], dtype=np.int32))

for i in range(len(neck_rings) - 1):
    tris.append(bridge(neck_rings[i], neck_rings[i + 1], closed=False))

T = np.vstack(tris)

# ── normalise: centre at origin, unit-ish scale, +Z forward ─────────────────
mn, mx = ALL.min(axis=0), ALL.max(axis=0)
centre = (mn + mx) / 2
ALL -= centre
ALL /= (mx - mn).max() / 2.0          # longest axis -> [-1, 1]

# ── wireframe edges (deduplicated) ──────────────────────────────────────────
eset = set()
for a, b, c in T:
    for e in ((a, b), (b, c), (c, a)):
        eset.add((min(e), max(e)))
E = np.array(sorted(eset), dtype=np.int32)

print(f"head mesh: {len(ALL)} verts, {len(T)} tris, {len(E)} edges")
assert len(ALL) < 65536, "index overflow for Uint16"

fmt = lambda a: ",".join(f"{x:.4f}".rstrip("0").rstrip(".") or "0" for x in a)

with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(f'''/**
 * headMesh.ts — AUTO-GENERATED, do not edit by hand.
 *
 * A complete human head built from MediaPipe's canonical face model
 * (468 vertices of real, anatomically-derived face topology — the same mesh
 * face-tracking systems use). The canonical model is an open face shell, so
 * its boundary loop is swept backwards over an ellipsoidal profile to close
 * the cranium, and a short neck is extruded from the jaw.
 *
 * Face data: github.com/google-ai-edge/mediapipe — Apache License 2.0.
 * Regenerate with scripts/build_head_mesh.py.
 *
 * Coordinates are centred on the origin and normalised so the longest axis
 * spans [-1, 1]. +Z points out of the face.
 */

/** Vertex positions, xyz triplets. */
export const HEAD_POSITIONS = new Float32Array([{fmt(ALL.ravel())}]);

/** Triangle indices (solid surface). */
export const HEAD_INDICES = new Uint16Array([{fmt(T.ravel())}]);

/** Deduplicated edge pairs (wireframe overlay). */
export const HEAD_EDGES = new Uint16Array([{fmt(E.ravel())}]);

/** Number of vertices belonging to the original MediaPipe face shell. */
export const FACE_VERTEX_COUNT = {n_face_verts};
''')
print("wrote", OUT)
