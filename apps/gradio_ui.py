import os
# Use an expandable allocator so PyTorch returns freed blocks to the driver, leaving
# room for nvblox's separate CUDA allocator to grow (helps avoid the OOM crash).
# Must be set BEFORE torch is imported (prism_vggt pulls it in below).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import tempfile
from PIL import Image

from prism_vggt.backends.panovggt import PanoVGGTBackend
from prism_vggt.engine import StreamingWindowEngine
from prism_vggt.utils.masking import get_spherical_valid_mask
from prism_vggt.utils.visualization import visualize_polar_mask, visualize_depth
from prism_vggt.utils.geometry import unproject_equirectangular_to_points

# --- Default configuration ------------------------------------------------------
# Single place to tweak the UI defaults / engine startup parameters.
CONFIG_DEFAULTS = {
    "step_size": 14,
    "target_width": 1036,
    "target_height": 518,
    "zenith_limit": 75,
    "nadir_limit": -70,
    "window_size": 16,
    "overlap": 4,
    "voxel_size": 0.02,
    "max_depth": 4.5,
    "camera_height": 1.7,
    "face_size": 768,
    "mesh_extract_every": 1,
    "sensor_mode": "lidar",
}

print("[UI] Initializing Architecture Stack...")
perception = PanoVGGTBackend(weights_path="checkpoints/model.pt")
streaming_engine = StreamingWindowEngine(
    perception=perception,
    voxel_size=CONFIG_DEFAULTS["voxel_size"],
    max_depth=CONFIG_DEFAULTS["max_depth"],
    target_camera_height=CONFIG_DEFAULTS["camera_height"],
    face_size=CONFIG_DEFAULTS["face_size"],
)

backend_state = {"frames": [], "mesh": None}

def get_o3d_pcd(xyz_points, rgb_image, mask):
    valid_mask = mask.astype(bool)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_points[valid_mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb_image[valid_mask] / 255.0)
    return pcd

def save_pcd_to_ply(pcd, prefix="reconstruction"):
    temp_dir = tempfile.mkdtemp()
    ply_path = os.path.join(temp_dir, f"{prefix}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    return ply_path

def save_mesh_to_glb(mesh, prefix="reconstruction"):
    if mesh is None or len(mesh.vertices) == 0: return None
    temp_dir = tempfile.mkdtemp()
    glb_path = os.path.join(temp_dir, f"{prefix}.glb")
    o3d.io.write_triangle_mesh(glb_path, mesh)
    return glb_path

def create_plotly_figure_from_pcd(pcd, max_points=150000):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    colors_str = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]
    fig = go.Figure(data=[go.Scatter3d(x=points[:, 0], y=points[:, 2], z=-points[:, 1], mode='markers', marker=dict(size=1.5, color=colors_str, opacity=1.0))])
    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="#111111")
    return fig

def add_ground_plane_trace(fig, plane, size_scale=1.5):
    """Render the exact detected floor plane as a semi-transparent quad.

    ``plane`` is the dict produced by the engine: world-frame ``normal``, ``centroid``
    and ``extent``. Axis remap matches the point cloud: plot x=X, y=Z, z=-Y.
    """
    if not plane:
        return
    n = np.asarray(plane["normal"], dtype=float)
    n = n / (np.linalg.norm(n) + 1e-12)
    c = np.asarray(plane["centroid"], dtype=float)
    ext = float(plane["extent"]) * size_scale

    # Build an orthonormal basis spanning the plane.
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, helper)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)

    corners = np.array([
        c - ext * u - ext * v,
        c + ext * u - ext * v,
        c + ext * u + ext * v,
        c - ext * u + ext * v,
    ])
    X, Y, Z = corners[:, 0], corners[:, 2], -corners[:, 1]
    fig.add_trace(go.Mesh3d(
        x=X, y=Y, z=Z,
        i=[0, 0], j=[1, 2], k=[2, 3],
        color="limegreen", opacity=0.35, name="Ground Plane",
        showlegend=True, hoverinfo="name"
    ))


def create_plotly_figure_with_trajectory(pcd, trajectory, plane=None, show_ground_plane=False, max_points=150000):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points, colors = points[idx], colors[idx]

    colors_str = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colors]

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 2], z=-points[:, 1], mode='markers', marker=dict(size=1.5, color=colors_str, opacity=1.0), name='Geometry'))

    if trajectory is not None and len(trajectory) > 0:
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0], y=trajectory[:, 2], z=-trajectory[:, 1],
            mode='lines+markers', name='Camera Trajectory',
            line=dict(color='cyan', width=4), marker=dict(size=4, color='orange')
        ))

    if show_ground_plane:
        add_ground_plane_trace(fig, plane)

    fig.update_layout(scene=dict(aspectmode='data', xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)), margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor="#111111", legend=dict(x=0.02, y=0.98, font=dict(color="white")))
    return fig

def process_single_frame(input_image_pil, zenith_limit, nadir_limit, target_width, target_height):
    if input_image_pil is None: return None, None, None, None
    input_image_pil = input_image_pil.resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS)
    input_image = np.array(input_image_pil)
    H, W = input_image.shape[:2]

    mask = get_spherical_valid_mask(H, W, zenith_deg=zenith_limit, nadir_deg=nadir_limit)
    masked_rgb_vis = visualize_polar_mask(input_image, mask)

    preds = perception.process_frame(input_image)
    depth_map = preds["depth"]

    xyz_points = unproject_equirectangular_to_points(np.squeeze(depth_map))
    depth_map[~mask] = 0.0
    depth_vis = visualize_depth(depth_map)

    pcd = get_o3d_pcd(xyz_points, input_image, mask)
    return Image.fromarray(masked_rgb_vis), Image.fromarray(depth_vis), create_plotly_figure_from_pcd(pcd), save_pcd_to_ply(pcd, "single_frame")

def get_file_list(input_mode, uploaded_files, local_dir, decimation):
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    if input_mode == "Upload Files" and uploaded_files:
        files = sorted([f.name for f in uploaded_files])
    elif input_mode != "Upload Files" and local_dir and os.path.isdir(local_dir):
        files = sorted([os.path.join(local_dir, f) for f in os.listdir(local_dir) if f.lower().endswith(valid_exts)])
    else:
        return []
    step = max(1, int(decimation) + 1)
    return files[::step]

def check_files_ui(input_mode, uploaded_files, local_dir, decimation):
    files = get_file_list(input_mode, uploaded_files, local_dir, decimation)
    if not files: return "⚠️ No valid images found."
    return f"✅ Total files to process: {len(files)}\n\n" + "\n".join(f"{i + 1}. {os.path.basename(n)}" for i, n in enumerate(files))

def process_sequence_ui(
    input_mode, uploaded_files, local_dir, decimation,
    zenith_limit, nadir_limit, target_width, target_height,
    window_size, overlap, max_depth, voxel_size, camera_height, face_size, mesh_extract_every,
    sensor_mode, live_stream_toggle, show_ground_plane
):
    file_paths = get_file_list(input_mode, uploaded_files, local_dir, decimation)
    if not file_paths or len(file_paths) < 2: raise gr.Error("Please provide at least 2 valid images.")

    frames, masks = [], []
    for path in file_paths:
        img_np = np.array(Image.open(path).convert("RGB").resize((int(target_width), int(target_height)), Image.Resampling.LANCZOS))
        frames.append(img_np)
        masks.append(get_spherical_valid_mask(img_np.shape[0], img_np.shape[1], zenith_deg=zenith_limit, nadir_deg=nadir_limit))

    backend_state["frames"] = frames

    streaming_engine.max_depth = float(max_depth)
    streaming_engine.voxel_size = float(voxel_size)
    streaming_engine.target_camera_height = float(camera_height)
    streaming_engine.face_size = int(face_size)
    streaming_engine.mesh_extract_every = int(mesh_extract_every)
    streaming_engine.sensor_mode = str(sensor_mode)

    last_mesh, last_pcd, last_traj, last_plane = None, None, None, None
    generator = streaming_engine.process_sequence(frames=frames, masks=masks, window_size=int(window_size), overlap=int(overlap))

    for mesh, global_pcd, trajectory, plane in generator:
        last_mesh, last_pcd, last_traj, last_plane = mesh, global_pcd, trajectory, plane
        if live_stream_toggle:
            fig = create_plotly_figure_with_trajectory(global_pcd, trajectory, plane, show_ground_plane)
            pcd_path = save_pcd_to_ply(global_pcd, "live_map")
            if mesh is None or len(mesh.vertices) == 0:
                yield fig, pcd_path, None, None
            else:
                backend_state["mesh"] = mesh
                mesh_path = save_mesh_to_glb(mesh, "final_scene")
                yield fig, pcd_path, mesh_path, mesh_path

    if last_pcd is not None:
        fig = create_plotly_figure_with_trajectory(last_pcd, last_traj, last_plane, show_ground_plane)
        pcd_path = save_pcd_to_ply(last_pcd, "live_map")
        if last_mesh is None or len(last_mesh.vertices) == 0:
            yield fig, pcd_path, None, None
        else:
            backend_state["mesh"] = last_mesh
            mesh_path = save_mesh_to_glb(last_mesh, "final_scene")
            yield fig, pcd_path, mesh_path, mesh_path

with gr.Blocks(theme=gr.themes.Monochrome(), title="PRISM-VGGT Streaming Sandbox") as demo:
    gr.Markdown("# 🌐 PRISM-VGGT: Alignment & Streaming Sandbox")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 Real-Time Execution")
            live_stream_checkbox = gr.Checkbox(value=False, label="Live UI Streaming (Slows down processing)")
            show_ground_plane_checkbox = gr.Checkbox(value=False, label="Show Detected Ground Plane")

            gr.Markdown("### Processing Controls")
            with gr.Row():
                step_size = gr.Number(value=CONFIG_DEFAULTS["step_size"], label="Step Size")
                link_ratio = gr.Checkbox(value=True, label="Link Aspect Ratio")

            target_width = gr.Slider(minimum=224, maximum=4096, value=CONFIG_DEFAULTS["target_width"], step=1, label="Target Width")
            target_height = gr.Slider(minimum=112, maximum=2048, value=CONFIG_DEFAULTS["target_height"], step=1, label="Target Height")

            gr.Markdown("### Polar Exclusion Limits")
            zenith_slider = gr.Slider(minimum=0, maximum=90, value=CONFIG_DEFAULTS["zenith_limit"], step=1, label="Zenith Limit")
            nadir_slider = gr.Slider(minimum=-90, maximum=0, value=CONFIG_DEFAULTS["nadir_limit"], step=1, label="Nadir Limit")

            gr.Markdown("### Submap Configuration (SLAM)")
            window_size_slider = gr.Slider(minimum=3, maximum=32, value=CONFIG_DEFAULTS["window_size"], step=1, label="Submap Window Size")
            overlap_slider = gr.Slider(minimum=2, maximum=8, value=CONFIG_DEFAULTS["overlap"], step=1, label="Submap Overlap")

            gr.Markdown("### 🧠 Nvblox Dense GPU Parameters")
            voxel_size_slider = gr.Slider(minimum=0.01, maximum=0.10, value=CONFIG_DEFAULTS["voxel_size"], step=0.01, label="Voxel Resolution (m) [Lower = Denser]")
            max_depth_slider = gr.Slider(minimum=2.0, maximum=15.0, value=CONFIG_DEFAULTS["max_depth"], step=0.5, label="Max Depth Ray Cutoff (m)")
            camera_height_slider = gr.Slider(minimum=0.1, maximum=3.0, value=CONFIG_DEFAULTS["camera_height"], step=0.1, label="Target Camera Height (m)")
            face_size_slider = gr.Slider(minimum=256, maximum=1536, value=CONFIG_DEFAULTS["face_size"], step=64, label="Cubemap Face Resolution (px) [Higher = sharper geometry, ~no gain above input width]")
            mesh_extract_slider = gr.Slider(minimum=1, maximum=10, value=CONFIG_DEFAULTS["mesh_extract_every"], step=1, label="Rebuild Mesh Every N Submaps [Higher = faster, mesh refreshes less often]")
            sensor_mode_radio = gr.Radio(choices=["lidar", "cubemap"], value=CONFIG_DEFAULTS["sensor_mode"], label="Depth Integration Sensor [lidar = 1 spherical frame, cubemap = 6 faces]")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("1. Single Frame Extract"):
                    input_img = gr.Image(label="Input Single 360° Image", type="pil")
                    run_single_btn = gr.Button("Extract Geometry", variant="primary")
                    with gr.Row():
                        output_rgb = gr.Image(label="Masked Input", type="pil")
                        output_depth = gr.Image(label="Depth Map", type="pil")
                    output_3d_single = gr.Plot(label="Single Frame Point Cloud")
                    download_single = gr.File(label="💾 Download Frame .ply")

                with gr.Tab("2. Multi-Frame 4D Stitching"):
                    input_mode = gr.Radio(choices=["Upload Files", "Local Directory Path"], value="Upload Files", label="Input Mode")
                    input_seq = gr.File(label="Upload Image Sequence", file_count="multiple", file_types=["image"], visible=True)
                    local_dir_input = gr.Textbox(label="Absolute Local Directory Path", visible=False)

                    with gr.Row():
                        decimation_input = gr.Number(value=0, label="Decimation (Skip N files)", precision=0)
                        check_files_btn = gr.Button("Check Files")

                    checked_files_output = gr.Textbox(label="Files to be Processed", interactive=False, lines=3)
                    run_seq_btn = gr.Button("Align & Stitch Sequence", variant="primary")

                    with gr.Tabs():
                        with gr.Tab("Real-Time Map"):
                            output_3d_seq = gr.Plot(label="Global Stitched Map")
                            download_seq = gr.File(label="💾 Download Global .ply")
                        with gr.Tab("Live Geometry (.glb)"):
                            output_mesh = gr.Model3D(label="Real-Time TSDF Mesh")
                            download_mesh = gr.File(label="💾 Download Mesh .glb")

    def enforce_res(w, h, step, link, trig):
        step = max(1, int(step))
        if link:
            if trig == 'w': h = w / 2.0
            elif trig == 'h': w = h * 2.0
        return int(round(w / step) * step), int(round(h / step) * step)

    target_width.release(fn=lambda w, h, s, l: enforce_res(w, h, s, l, 'w'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height])
    target_height.release(fn=lambda w, h, s, l: enforce_res(w, h, s, l, 'h'), inputs=[target_width, target_height, step_size, link_ratio], outputs=[target_width, target_height])

    input_mode.change(fn=lambda m: (gr.update(visible=m=="Upload Files"), gr.update(visible=m!="Upload Files")), inputs=input_mode, outputs=[input_seq, local_dir_input])
    check_files_btn.click(fn=check_files_ui, inputs=[input_mode, input_seq, local_dir_input, decimation_input], outputs=[checked_files_output], api_name=False)
    run_single_btn.click(fn=process_single_frame, inputs=[input_img, zenith_slider, nadir_slider, target_width, target_height], outputs=[output_rgb, output_depth, output_3d_single, download_single])

    run_seq_btn.click(
        fn=process_sequence_ui,
        inputs=[
            input_mode, input_seq, local_dir_input, decimation_input, zenith_slider, nadir_slider,
            target_width, target_height, window_size_slider, overlap_slider,
            max_depth_slider, voxel_size_slider, camera_height_slider, face_size_slider, mesh_extract_slider,
            sensor_mode_radio, live_stream_checkbox, show_ground_plane_checkbox
        ],
        outputs=[output_3d_seq, download_seq, output_mesh, download_mesh]
    )

if __name__ == "__main__":
    # Entry point for the PRISM-VGGT streaming sandbox.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
