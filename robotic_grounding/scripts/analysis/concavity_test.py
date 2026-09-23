"""Is the convex hull a good proxy for this solid? Measure it without needing watertightness."""
import trimesh, numpy as np, time

def hull_gap_fraction(mesh, n=20000, tol=0.01, seed=0):
    """Fraction of the convex hull's surface that sits more than `tol` from the mesh.

    Convex solid -> hull hugs the surface -> ~0.  Mug / open frame -> the hull caps
    the opening, and those caps are far from any real surface -> large.
    Independent of watertightness and of any voxel pitch.
    """
    hull = mesh.convex_hull
    pts, _ = trimesh.sample.sample_surface_even(hull, n, seed=seed)
    if len(pts) == 0:
        return float("nan")
    d = trimesh.proximity.ProximityQuery(mesh).signed_distance(pts)
    return float((np.abs(d) > tol).mean())

mp='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/tissue_box_refined.obj'
cases = [("tissue_box (reconstructed, NOT watertight)", trimesh.load(mp, force="mesh"))]
box = trimesh.creation.box(extents=[0.2,0.15,0.15]); cases.append(("box (watertight, convex)", box))
cup = trimesh.creation.cylinder(radius=0.05, height=0.12, sections=48)
inner = trimesh.creation.cylinder(radius=0.043, height=0.115, sections=48); inner.apply_translation([0,0,0.012])
cases.append(("mug-like (hollow cylinder)", trimesh.boolean.difference([cup, inner])))
shell = trimesh.load(mp, force="mesh").copy()

print(f"{'object':<42} {'watertight':>10} {'vol ratio':>10} {'hull gap':>9}  {'time':>6}")
for name, m in cases:
    t=time.time()
    try:
        vr = m.convex_hull.volume/m.volume if m.volume and m.volume>1e-10 else float('inf')
    except Exception:
        vr = float('nan')
    g = hull_gap_fraction(m)
    print(f"{name:<42} {str(m.is_watertight):>10} {vr:>10.3f} {g:>8.3f}  {time.time()-t:5.1f}s")
