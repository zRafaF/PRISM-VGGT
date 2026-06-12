import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

def load_npz_node(file_path):
    print(f"📦 Loading local node: {os.path.basename(file_path)}")
    data = np.load(file_path)
    points = data['points'].reshape(-1, 3)
    colors = data['colors'].reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def load_global_map(file_path):
    print(f"🌍 Loading global map: {os.path.basename(file_path)}")
    return o3d.io.read_point_cloud(file_path)

def load_glb_model(file_path):
    print(f"🧊 Loading 3D model: {os.path.basename(file_path)}")
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    if not mesh.has_vertices():
        print("⚠️ Open3D failed to parse GLB. Falling back to Trimesh...")
        import trimesh
        scene_or_mesh = trimesh.load(file_path, force='mesh')
        
        geom = scene_or_mesh.dump(concatenate=True) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh
            
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(geom.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(geom.faces)
        
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            colors = geom.visual.vertex_colors[:, :3] / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        
    return mesh

def launch_measurement_tool(geometry):
    print("\n" + "="*50)
    print("📏 LAUNCHING MEASUREMENT TOOL")
    print("="*50)
    print("  1. Hold [SHIFT] and [LEFT CLICK] on TWO different points.")
    print("  2. The distance between them will be calculated automatically.")
    print("  3. Press [Q] or close the window to exit the measurement tool.")
    print("="*50 + "\n")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(title="PRISM Measurement Tool (SHIFT + Click)", width=1280, height=720)
    vis.add_geometry(geometry)
    vis.run() 
    vis.destroy_window()

    picked_indices = vis.get_picked_points()
    if len(picked_indices) >= 2:
        # Determine if we are measuring a Mesh or a Point Cloud
        points_array = np.asarray(geometry.vertices) if isinstance(geometry, o3d.geometry.TriangleMesh) else np.asarray(geometry.points)
        
        p1 = points_array[picked_indices[0]]
        p2 = points_array[picked_indices[1]]
        dist_m = np.linalg.norm(p1 - p2)
        dist_cm = dist_m * 100.0
        
        print("\n✅ MEASUREMENT COMPLETE:")
        print(f"   -> Point 1: {p1}")
        print(f"   -> Point 2: {p2}")
        print(f"   -> Distance: {dist_m:.3f} Meters ({dist_cm:.1f} cm)\n")
    else:
        print("\n⚠️ Not enough points selected to calculate distance. You must SHIFT+Click at least two points.\n")

def main():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)

    print("📂 Please select a Point Cloud or 3D Model file to open...")
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select 3D File",
        filetypes=[
            ("All Supported Files", "*.npz *.ply *.glb"),
            ("Single Node (.npz)", "*.npz"),
            ("Global Map (.ply)", "*.ply"),
            ("3D Model (.glb)", "*.glb")
        ]
    )
    
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npz':
        geom = load_npz_node(file_path)
    elif ext == '.ply':
        geom = load_global_map(file_path)
    elif ext == '.glb':
        geom = load_glb_model(file_path)
    else:
        return

    # 1. Launch the new Measurement App Tool
    launch_measurement_tool(geom)
    
    # 2. Launch the standard high-performance viewer afterward
    print("🚀 Launching Standard 3D Viewer...")
    print("   -> Change 'Mouse control' from 'Arcball' to 'Fly' in the UI panel.")
    o3d.visualization.draw([{"name": "Reconstruction", "geometry": geom}], title="PRISM 3D Viewer", bg_color=(0.05, 0.05, 0.05, 1.0), show_ui=True)

if __name__ == "__main__":
    main()