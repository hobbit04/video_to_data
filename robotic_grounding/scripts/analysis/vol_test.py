import trimesh, numpy as np, time
mp='/workspace/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/tissue_box_refined.obj'
m=trimesh.load(mp, force="mesh")
hull=m.convex_hull
print(f"watertight={m.is_watertight}  faces={len(m.faces)}  extents={np.round(m.extents,4)}")
print(f"mesh.volume      = {m.volume:.6f}   -> ratio {hull.volume/m.volume:7.3f}")
print(f"hull.volume      = {hull.volume:.6f}")
print(f"OBB volume       = {m.bounding_box_oriented.volume:.6f}")
r=m.copy(); trimesh.repair.fill_holes(r)
print(f"fill_holes: watertight={r.is_watertight} volume={r.volume:.6f}"
      + (f" -> ratio {hull.volume/r.volume:7.3f}" if r.is_watertight and r.volume>1e-10 else ""))
for div in (32, 48, 64):
    t=time.time(); pitch=float(m.extents.max())/div
    try:
        v=float(m.voxelized(pitch=pitch).fill().volume)
        print(f"voxel div={div:3d} pitch={pitch*100:5.2f}cm  volume={v:.6f}  ratio {hull.volume/v:7.3f}  ({time.time()-t:.2f}s)")
    except Exception as e:
        print(f"voxel div={div}: FAILED {e}")
