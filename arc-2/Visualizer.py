import bpy
import json
import os

# =========================================================================
# 1. SETUP & CLEAR SCENE
# =========================================================================
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

for mat in bpy.data.materials:
    bpy.data.materials.remove(mat)

# =========================================================================
# 2. ARC COLOR PALETTE & FLAT MATERIALS
# =========================================================================
arc_colors_rgb = {
    0: (0.1, 0.1, 0.1),
    1: (0.00, 0.45, 0.85),
    2: (1.00, 0.25, 0.21),
    3: (0.18, 0.80, 0.25),
    4: (1.00, 0.86, 0.00),
    5: (0.67, 0.67, 0.67),
    6: (0.94, 0.07, 0.74),
    7: (1.00, 0.52, 0.10),
    8: (0.50, 0.86, 1.00),
    9: (0.53, 0.05, 0.14),
}

materials = {}
for color_id, rgb in arc_colors_rgb.items():
    mat = bpy.data.materials.new(name=f"ARC_{color_id}")
    tree = mat.node_tree if mat.node_tree else mat.ensure_node_tree()
    nodes = tree.nodes
    links = tree.links
    
    for n in nodes: 
        nodes.remove(n)
    
    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    emit_node = nodes.new(type='ShaderNodeEmission')
    
    emit_node.inputs['Color'].default_value = (*rgb, 1.0)
    emit_node.inputs['Strength'].default_value = 1.5 
    
    links.new(emit_node.outputs['Emission'], out_node.inputs['Surface'])
    materials[color_id] = mat

# =========================================================================
# 3. LOAD MCTS PATH DATA (Happy Path First + Debugging)
# =========================================================================
# '//' tells Blender to look in the exact folder where the .blend file is saved
json_path = bpy.path.abspath("//mcts_path.json")

# DEBUG: Let's force Blender to tell us exactly where it is looking!
print("="*50)
print(f"DEBUG: Blender is looking for the file at this exact path:")
print(f"DEBUG: {json_path}")
print(f"DEBUG: Does the file exist at this path? {os.path.exists(json_path)}")
print("="*50)

if os.path.exists(json_path):
    # HAPPY PATH: The file exists! Read it.
    with open(json_path, 'r') as f: 
        sequence = json.load(f)
    print(f"SUCCESS: Loaded {len(sequence)} frames from mcts_path.json")
else:
    # FALLBACK PATH: The file is missing. Use dummy data.
    sequence = [
        [[0,0,0,0], [0,2,0,0], [0,0,0,0], [0,0,0,0]], 
        [[0,0,0,0], [0,0,0,0], [0,2,0,0], [0,0,0,0]], 
        [[0,0,0,0], [0,0,0,0], [0,0,2,0], [0,0,0,0]], 
        [[0,0,0,0], [0,0,0,0], [0,0,0,2], [0,0,0,0]], 
    ]
    print("WARNING: mcts_path.json NOT FOUND. Using dummy data.")

grid_h = len(sequence[0])
grid_w = len(sequence[0][0])

# =========================================================================
# 4. GENERATE VOXEL GRID
# =========================================================================
cubes = []
for r in range(grid_h):
    row_cubes = []
    for c in range(grid_w):
        bpy.ops.mesh.primitive_cube_add(size=0.95, location=(c, -r, 0))
        obj = bpy.context.active_object
        obj.scale[2] = 0.1
        obj.name = f"Tile_R{r}_C{c}"
        row_cubes.append(obj)
    cubes.append(row_cubes)

# =========================================================================
# 5. KEYFRAME ANIMATION (DISCRETE SNAPPING)
# =========================================================================
frames_per_step = 20 
bpy.context.scene.frame_start = 0
bpy.context.scene.frame_end = len(sequence) * frames_per_step

for step_idx, grid_state in enumerate(sequence):
    frame_num = step_idx * frames_per_step
    bpy.context.scene.frame_set(frame_num)
    
    for r in range(grid_h):
        for c in range(grid_w):
            color_val = grid_state[r][c]
            obj = cubes[r][c]
            
            if obj.data.materials:
                obj.data.materials[0] = materials[color_val]
            else:
                obj.data.materials.append(materials[color_val])
                
            if color_val == 0:
                obj.location[2] = 0.0
            else:
                obj.location[2] = 0.4
                
            obj.keyframe_insert(data_path="location", frame=frame_num)

# Fix for Blender 4.2+ / 5.x Layer-based animation API
for r in range(grid_h):
    for c in range(grid_w):
        obj = cubes[r][c]
        if obj.animation_data and obj.animation_data.action:
            try:
                for fcurve in obj.animation_data.action.fcurves:
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = 'CONSTANT'
            except AttributeError:
                try:
                    for layer in obj.animation_data.action.layers:
                        for strip in layer.strips:
                            if hasattr(strip, 'fcurves'):
                                for fcurve in strip.fcurves:
                                    for kf in fcurve.keyframe_points:
                                        kf.interpolation = 'CONSTANT'
                except Exception:
                    pass

# =========================================================================
# 6. CAMERA, LIGHTING & VIEWPORT SNAP
# =========================================================================
bpy.ops.object.camera_add(location=(grid_w/2 - 0.5, -grid_h/2 + 0.5, 15))
cam = bpy.context.active_object
cam.rotation_euler = (0, 0, 0)
cam.data.type = 'ORTHO'
cam.data.ortho_scale = max(grid_w, grid_h) + 8
bpy.context.scene.camera = cam

bpy.ops.object.light_add(type='SUN', location=(0, 0, 15))
sun = bpy.context.active_object
sun.data.energy = 1.0

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        area.spaces[0].region_3d.view_perspective = 'CAMERA'

bpy.context.scene.frame_set(0)
print("Visualizer Ready. Press Spacebar to play animation.")