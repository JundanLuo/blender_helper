import os
from typing import List, Optional
import importlib
import warnings

import bpy, bmesh
import mathutils
import numpy as np


from blender_helper.blender_utils import constants

importlib.reload(constants)
C = constants


print("Helper module loaded")


class CameraParams:
    def __init__(self, type, fov, fov_axis, loc, rot, img_width, img_height,
                 sensor_width, sensor_height):
        self.type = type
        self.fov = fov
        self.fov_axis = fov_axis
        self.loc = loc
        self.rot = rot
        self.img_width = img_width
        self.img_height = img_height
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

    def to_dict(self):
        return {
            "type": self.type,
            "fov": self.fov,
            "fov_axis": self.fov_axis,
            "location": self.loc.tolist(),
            "rotation": self.rot.tolist(),
            "img_width": self.img_width,
            "img_height": self.img_height,
            "sensor_width": self.sensor_width,
            "sensor_height": self.sensor_height
        }


def blender_version():
    return bpy.app.version_string


def find_scene(scene_name):
    return bpy.data.scenes.get(scene_name)


def get_current_scene_name():
    return bpy.context.scene.name


def activate_scene(scene_name):
    assert find_scene(scene_name) is not None, f"Scene {scene_name} not found"
    bpy.context.window.scene = find_scene(scene_name)


def copy_scene(scene_name, new_scene_name, is_link):
    assert find_scene(scene_name) is not None, f"Scene {scene_name} not found"
    activate_scene(scene_name)  # activate the source scene
    if is_link:
        bpy.ops.scene.new(type='LINK_COPY')  # create a linked copy
    else:
        bpy.ops.scene.new(type='FULL_COPY')  # create a full copy
    bpy.context.scene.name = new_scene_name  # rename the new scene


def remove_scene(scene_name):
    scene = find_scene(scene_name)
    if scene is None:
        print(f"Scene {scene_name} not found")
        return
    bpy.data.scenes.remove(scene)


def find_object(name):
    return bpy.data.objects.get(name)


def find_collection(name):
    return bpy.data.collections.get(name)


def configure_render_engine(engine='CYCLES', sample_count=128, min_samples=30, time_limit=10, max_bounces=24,
                            use_gpu=True,
                            transparent_bg=False):
    """
    Configure the rendering engine.
    Note that set transparent_bg=False if you need to render an environmental light map.
    :param engine:
    :param sample_count:
    :param min_samples:
    :param time_limit:
    :param max_bounces:
    :param use_gpu:
    :param transparent_bg:
    :return:
    """

    assert engine == 'CYCLES', "Only Cycles rendering engine is supported"

    # Set the rendering engine
    bpy.context.scene.render.engine = engine
    bpy.context.scene.cycles.feature_set = 'SUPPORTED'

    # Set the number of view samples
    bpy.context.scene.cycles.preview_samples = 32

    # Set the number of render samples
    bpy.context.scene.cycles.samples = sample_count
    bpy.context.scene.cycles.adaptive_min_samples = min_samples
    bpy.context.scene.cycles.time_limit = time_limit

    # Denoising
    bpy.context.scene.cycles.use_denoising = True
    if use_gpu:
        bpy.context.scene.cycles.denoiser = 'OPTIX'
    else:
        bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    bpy.context.scene.cycles.use_adaptive_sampling = True
    bpy.context.scene.cycles.adaptive_threshold = 0.0001

    # Set the number of bounces
    ratio = max_bounces / 24.0
    bpy.context.scene.cycles.max_bounces = int(24 * ratio)
    bpy.context.scene.cycles.diffuse_bounces = int(16 * ratio)
    bpy.context.scene.cycles.glossy_bounces = int(8 * ratio)
    bpy.context.scene.cycles.transmission_bounces = int(16 * ratio)
    bpy.context.scene.cycles.volume_bounces = int(2 * ratio)
    bpy.context.scene.cycles.transparent_max_bounces = int(16 * ratio)

    # Set the background to transparent or not
    bpy.context.scene.render.film_transparent = transparent_bg

    # Select the backend explicitly; a fresh Blender installation defaults to NONE.
    if use_gpu:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'OPTIX'
        preferences.get_devices()
        gpu_devices = [device for device in preferences.devices if device.type == 'OPTIX']
        if not gpu_devices:
            raise RuntimeError('GPU rendering requires an available OptiX device.')
        for device in preferences.devices:
            device.use = device.type == 'OPTIX'
        print('OptiX rendering devices:', ', '.join(device.name for device in gpu_devices))
        bpy.context.scene.cycles.device = 'GPU'
    else:
        bpy.context.scene.cycles.device = 'CPU'

    # Color management
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.view_settings.look = 'None'
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1
    bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'

    # Print configuration to console
    print(
        f"{engine} configured: {sample_count} samples, GPU enabled: {use_gpu}"
    )


def configure_output(output_path, update_camera_sensor_for_square_pixel: bool, img_width=640, img_height=480, output_format='PNG'):
    """
    Configure the output settings.
    Note the camera sensor may need to be updated after setting the resolution,
    use update_camera_sensor_for_square_pixel=True to let fx == fy.
    """

    # Set the output path
    bpy.context.scene.render.filepath = output_path

    # Set the resolution
    bpy.context.scene.render.resolution_x = img_width
    bpy.context.scene.render.resolution_y = img_height
    if update_camera_sensor_for_square_pixel:
        camera = bpy.context.scene.camera
        if camera is not None and camera.data.type == "PERSP":
            update_camera_sensor_size_for_square_pixel(camera.name)
        else:
            print("Camera not found or not in perspective mode")

    # Set the output format
    bpy.context.scene.render.image_settings.file_format = output_format
    if output_format == "PNG":
        bpy.context.scene.render.image_settings.color_mode = "RGB"
        bpy.context.scene.render.image_settings.color_depth = "8"
        bpy.context.scene.render.image_settings.compression = 0
        bpy.context.scene.render.image_settings.color_management = 'OVERRIDE'
        bpy.context.scene.display_settings.display_device = 'sRGB'
        bpy.context.scene.view_settings.view_transform = 'Standard'
        bpy.context.scene.view_settings.look = 'None'
        bpy.context.scene.view_settings.exposure = 0
        bpy.context.scene.view_settings.gamma = 1
    elif output_format == "OPEN_EXR":
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.color_depth = "32"
        bpy.context.scene.render.image_settings.exr_codec = 'ZIP'
        bpy.context.scene.render.image_settings.color_management = 'OVERRIDE'
        # bpy.context.scene.colorspace_settings.name = 'Linear'
    else:
        print(f"Unsupported output format: {output_format}")

    # Print configuration to console
    print(f"Output configured: {output_path}, format: {output_format}")


def setup_render_passes(clear_all=False, set_active_layers: List[C.LayerStrMapping] = None):
    if bpy.context.scene.render.engine != 'CYCLES':
        print("Only Cycles rendering engine is supported for render passes setup.")
        return

    # Access the active render layer
    render_layer = bpy.context.view_layer
    render_layer.use = True

    # Clear all passes
    if clear_all:
        for attr in dir(render_layer):
            if attr.startswith("use_pass_"):
                setattr(render_layer, attr, False)
        bpy.context.view_layer.cycles.use_pass_shadow_catcher = False
        render_layer.use_pass_combined = True

    # Set active passes
    if set_active_layers is None:
        return
    for p in set_active_layers:
        if p.pass_name == "mask":
            continue
        n = f"use_pass_{p.pass_name}"
        if not hasattr(render_layer, n):
            if p != C.mask:
                print(f"Render pass {p.pass_name} not found")
            continue
        setattr(render_layer, n, True)
    if C.shadow_catcher in set_active_layers:
        bpy.context.view_layer.cycles.use_pass_shadow_catcher = True


def insert_keyframe(obj_name, data_path, frame_idx):
    obj = find_object(obj_name)
    assert obj is not None, f"Object {obj_name} not found"
    if isinstance(data_path, str):
        data_path = [data_path]
    for dp in data_path:
        obj.keyframe_insert(data_path=dp, frame=frame_idx)


def clear_keyframes(obj_name):
    obj = find_object(obj_name)
    assert obj is not None, f"Object {obj_name} not found"
    obj.animation_data_clear()


def set_fps(fps):
    bpy.context.scene.render.fps = fps
    bpy.context.scene.render.fps_base = 1


def set_frame_range(start_frame, end_frame):
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame


def render_scene(frame_idx=None, write_still=False, render_animation=False):
    if frame_idx is not None:
        assert isinstance(frame_idx, int), "frame_idx should be an integer"
        bpy.context.scene.frame_set(frame_idx)
    bpy.ops.render.render(write_still=write_still,
                          animation=render_animation)


def update_camera_sensor_size_for_square_pixel(camera_name):
    """
    This function adjusts the sensor size to let fx == fy.
    Refer to: https://gist.github.com/autosquid/8e1cddbc0336a49c6f84591d35371c4d
    https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    :param camera_name:
    :return:
    """
    # Get camera object
    camera = find_object(camera_name)
    assert camera is not None, f"Camera {camera_name} not found"
    fov = camera.data.angle
    # Check sensor fit
    sensor_fit = camera.data.sensor_fit
    assert sensor_fit in ["HORIZONTAL"], f"Unsupported sensor_fit: {sensor_fit}"
    # Check camera type
    assert camera.data.type == "PERSP", f"Unsupported camera type: {camera.data.type}"
    # Adjust sensor size to let fx == fy
    pixel_aspect_ratio = bpy.context.scene.render.pixel_aspect_x / bpy.context.scene.render.pixel_aspect_y
    assert pixel_aspect_ratio == 1, f"Not support pixel_aspect_ratio != 1: {pixel_aspect_ratio}"
    render_resolution_x = bpy.context.scene.render.resolution_x
    render_resolution_y = bpy.context.scene.render.resolution_y
    camera.data.sensor_height = render_resolution_y / render_resolution_x * camera.data.sensor_width
    # Check
    sensor_ratio = camera.data.sensor_width / camera.data.sensor_height
    resolution_ratio = render_resolution_x / render_resolution_y
    assert abs(sensor_ratio - resolution_ratio) < 1e-6, (f"sensor_ratio != resolution_ratio: "
                                                         f"{sensor_ratio} != {resolution_ratio}")
    assert abs(camera.data.angle - fov) < 1e-6, f"camera's fov changed after sensor size adjustment: {camera.data.angle} != {fov}"


def setup_render_camera(camera_name, type=None, fov_deg=None, fov_axis=None,
                        clip=None, loc=None, rot_deg=None, is_render_camera=True):
    # Get camera object
    camera = find_object(camera_name)
    if camera is None:
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW',
                                  location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1))
        camera = bpy.context.object
        camera.name = camera_name
        print(f"Camera {camera_name} added.")
    set_object_visibility(camera_name, hide_render=False, hide_viewport=False)
    if is_render_camera:
        bpy.context.scene.camera = camera
    # Set type
    if type is not None:
        camera.data.type = type
    # Set FOV or angles
    if camera.data.type == "PERSP":
        camera.data.lens_unit = 'FOV'
        if fov_deg is not None:
            camera.data.angle = np.deg2rad(fov_deg)
        if fov_axis is not None:
            fov_axis = fov_axis.upper()
            assert fov_axis in ["HORIZONTAL"], f"Not support fov_axis {fov_axis}"
            camera.data.sensor_fit = fov_axis
        else:
            assert False, f"fov_axis is recommended to be specified. Comment out this line if you want to proceed."
        # Adjust sensor size to let fx == fy
        update_camera_sensor_size_for_square_pixel(camera_name)
        # Set clip start and end
        if clip is not None:
            camera.data.clip_start = clip[0]
            camera.data.clip_end = clip[1]
    elif camera.data.type == "PANO":
        camera.data.cycles.panorama_type = 'EQUIRECTANGULAR'
        camera.data.cycles.latitude_min = np.deg2rad(-90)
        camera.data.cycles.latitude_max = np.deg2rad(90)
        camera.data.cycles.longitude_min = np.deg2rad(-180)
        camera.data.cycles.longitude_max = np.deg2rad(180)
        camera.data.shift_x = 0
        camera.data.shift_y = 0
        camera.data.clip_start = 0.001
        camera.data.clip_end = 1000
    elif camera.data.type == "ORTHO":
        if clip is not None:
            camera.data.clip_start = clip[0]
            camera.data.clip_end = clip[1]
    else:
        print(f"Camera type {camera.data.type} not supported")
        return

    # Set camera location
    if loc is not None:
        if len(loc) != 3:
            print(f"Invalid location {loc}")
            return
        camera.location = loc
    # Set camera rotation
    if rot_deg is not None:
        if len(rot_deg) != 3:
            print(f"Invalid rotation {rot_deg}")
            return
        rot_rad = np.deg2rad(rot_deg)
        camera.rotation_euler = rot_rad


def add_output_node(tree, label=None, base_path=None, file_format=None, color_depth=None, location=None):
    assert tree is not None, "Tree is None"
    output_node = tree.nodes.new('CompositorNodeOutputFile')
    if label is not None:
        output_node.label = label
    if base_path is not None:
        output_node.base_path = base_path
    if file_format is not None:
        output_node.format.file_format = file_format
    if color_depth is not None:
        output_node.format.color_depth = color_depth
    if location is not None:
        output_node.location = location
    return output_node


def configure_composite(output_path, out_layers: List[C.LayerStrMapping] = None, object_with_mask=None):
    """
    Configure the composite nodes for rendering.
    :param output_path:
    :param out_layers:
    :param object_with_mask:
        the object name for which the mask is generated
    :return:
    """
    # Set up render passes
    setup_render_passes(clear_all=True, set_active_layers=out_layers)

    # Enable compositing
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create rendered layer node
    rlayers_node = tree.nodes.new('CompositorNodeRLayers')
    rlayers_node.location = 0, 0

    # Create output file node
    outfile_node_label_dict = {}
    out_paths = {}
    out_pass_dir = os.path.join(output_path, "passes")
    loc_x, loc_y = 400, -200
    for l in out_layers:
        outfile_node = tree.nodes.new('CompositorNodeOutputFile')
        outfile_node.label = f"{l.pass_name} output"
        outfile_node.base_path = os.path.join(out_pass_dir, l.pass_name)
        outfile_node.format.file_format = 'OPEN_EXR'
        outfile_node.format.color_depth = '32'
        outfile_node.location = loc_x, loc_y
        outfile_node_label_dict[l.pass_name] = outfile_node.label
        out_paths[l.pass_name] = outfile_node.base_path
        if l.rendered_name is not None:
            links.new(rlayers_node.outputs[l.rendered_name], outfile_node.inputs['Image'])
        if l.pass_name == C.object_mask.pass_name:
            if object_with_mask is not None:
                # add cryptomatte node. output the mask for the object
                cryptomatte_node = tree.nodes.new(type="CompositorNodeCryptomatteV2")
                active_layer = bpy.context.view_layer  # Note that the layer name is different in terms of the Blender version
                cryptomatte_node.layer_name = f'{active_layer.name}.CryptoObject'
                cryptomatte_node.matte_id = object_with_mask
                cryptomatte_node.location = loc_x - 400, loc_y
                links.new(cryptomatte_node.outputs['Matte'], outfile_node.inputs['Image'])
        loc_y += 200
    del outfile_node

    # Output combined shading
    loc_x += 200
    if C.diffuse_direct_shading in out_layers and C.diffuse_indirect_shading in out_layers:
        # add mix node
        mix_node = tree.nodes.new('CompositorNodeMixRGB')
        mix_node.blend_type = 'ADD'
        mix_node.location = loc_x, 0
        links.new(rlayers_node.outputs[C.diffuse_direct_shading.rendered_name], mix_node.inputs[1])
        links.new(rlayers_node.outputs[C.diffuse_indirect_shading.rendered_name], mix_node.inputs[2])
        # add denoise node
        loc_x += 200
        denoise_node = tree.nodes.new('CompositorNodeDenoise')
        denoise_node.prefilter = "ACCURATE"
        denoise_node.location = loc_x, 0
        links.new(mix_node.outputs['Image'], denoise_node.inputs['Image'])
        # add output node
        loc_x += 200
        outfile_node = tree.nodes.new('CompositorNodeOutputFile')
        outfile_node.label = "diffuse_shading output"
        outfile_node.base_path = os.path.join(out_pass_dir, "diffuse_shading")
        outfile_node.format.file_format = 'OPEN_EXR'
        outfile_node.format.color_depth = '32'
        outfile_node.location = loc_x, 0
        outfile_node_label_dict["diffuse_shading"] = outfile_node.label
        out_paths["diffuse_shading"] = outfile_node.base_path
        links.new(denoise_node.outputs['Image'], outfile_node.inputs['Image'])
    return outfile_node_label_dict, out_paths


def clear_composite_nodes():
    # Enable compositing
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)


def configure_composite_AR(foreground_scene_name, mask_scene_name, output_path, camera_name, with_background):
    """
    Configure the composite nodes for rendering.
    :param foreground_scene_name: the scene for rendering the foreground object
    :param mask_scene_name: the scene for rendering the mask
    :param output_path:
    :param camera_name:
    :param with_background:
    :return:
    """
    assert find_scene(foreground_scene_name) is not None, f"Scene {foreground_scene_name} not found"
    assert find_scene(mask_scene_name) is not None, f"Scene {mask_scene_name} not found"
    # Set up render passes
    activate_scene(foreground_scene_name)
    out_layers = [
        C.composite_image, C.diffuse_albedo,
        C.diffuse_direct_shading, C.diffuse_indirect_shading,
        C.normal, C.mask]
    setup_render_passes(clear_all=True, set_active_layers=out_layers)
    activate_scene(mask_scene_name)
    setup_render_passes(clear_all=True, set_active_layers=[
        C.composite_image, C.mask, C.shadow_catcher])
    activate_scene(foreground_scene_name)  # the major scene
    out_paths = {}

    # Enable compositing and clear default nodes
    clear_composite_nodes()
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Create rendered layer node
    rlayers_node = tree.nodes.new('CompositorNodeRLayers')
    rlayers_node.location = 0, 0
    rlayers_node.scene = find_scene(foreground_scene_name)
    rlayers_mask_scene_node = tree.nodes.new('CompositorNodeRLayers')
    rlayers_mask_scene_node.location = 0, -500
    rlayers_mask_scene_node.scene = find_scene(mask_scene_name)

    # Output object mask
    outfile_node = add_output_node(tree, "object mask output",
                                   os.path.join(output_path, "passes", "object_mask"),
                                   'OPEN_EXR', '32', (400, -700))
    out_paths["object_mask"] = outfile_node.base_path
    links.new(rlayers_mask_scene_node.outputs[C.mask.rendered_name], outfile_node.inputs['Image'])

    # Output shadow mask
    invert_node = tree.nodes.new('CompositorNodeInvert')  # shadow mask = 1.0 - shadow catcher
    invert_node.location = 400, -900
    invert_node.invert_rgb = True
    invert_node.invert_alpha = False
    links.new(rlayers_mask_scene_node.outputs[C.shadow_catcher.rendered_name], invert_node.inputs['Color'])
    multiply_node = tree.nodes.new('CompositorNodeMath')  # set too small values to zero
    multiply_node.operation = 'MULTIPLY'
    multiply_node.use_clamp = True
    multiply_node.location = 600, -900
    threshold_node = tree.nodes.new('CompositorNodeMath')
    threshold_node.operation = 'GREATER_THAN'
    threshold_node.location = 600, -1100
    threshold_node.use_clamp = True
    threshold_node.inputs[1].default_value = 0.05
    links.new(invert_node.outputs['Color'], threshold_node.inputs['Value'])
    links.new(invert_node.outputs['Color'], multiply_node.inputs[0])
    links.new(threshold_node.outputs['Value'], multiply_node.inputs[1])
    outfile_node = add_output_node(tree, "shadow mask output",
                                   os.path.join(output_path, "passes", "shadow_mask"),
                                   'OPEN_EXR', '32', (800, -900))
    out_paths["shadow_mask"] = outfile_node.base_path
    links.new(multiply_node.outputs['Value'], outfile_node.inputs['Image'])

    # Output full object mask
    full_mask_node = tree.nodes.new('CompositorNodeMath')
    full_mask_node.operation = 'MAXIMUM'
    full_mask_node.location = 600, -650
    full_mask_node.use_clamp = True
    links.new(rlayers_mask_scene_node.outputs[C.mask.rendered_name], full_mask_node.inputs[0])
    links.new(multiply_node.outputs['Value'], full_mask_node.inputs[1])
    outfile_node = add_output_node(tree, "full object mask output",
                                   os.path.join(output_path, "passes", "full_object_mask"),
                                   'OPEN_EXR', '32', (800, -650))
    out_paths["full_object_mask"] = outfile_node.base_path
    links.new(full_mask_node.outputs['Value'], outfile_node.inputs['Image'])

    # Create output file node
    out_pass_dir = os.path.join(output_path, "passes")
    loc_x, loc_y = 400, -200
    for l in out_layers:
        outfile_node = tree.nodes.new('CompositorNodeOutputFile')
        outfile_node.label = f"{l.rendered_name} output"
        outfile_node.base_path = os.path.join(out_pass_dir, l.pass_name)
        outfile_node.format.file_format = 'OPEN_EXR'
        outfile_node.format.color_depth = '32'
        outfile_node.location = loc_x, loc_y
        out_paths[l.pass_name] = outfile_node.base_path
        loc_y += 200
        links.new(rlayers_node.outputs[l.rendered_name], outfile_node.inputs['Image'])
        if l == C.composite_image:
            # add background
            if with_background:
                nloc_x, nloc_y = loc_x+100, loc_y
                # set output node
                outfile_node = tree.nodes.new('CompositorNodeOutputFile')
                outfile_node.label = "final image output"
                outfile_node.base_path = os.path.join(out_pass_dir, "final_image")
                outfile_node.format.file_format = 'PNG'
                outfile_node.format.color_depth = '8'
                outfile_node.location = nloc_x + 500, nloc_y
                out_paths["final_image"] = outfile_node.base_path
                # add mix node
                mix_node = tree.nodes.new('CompositorNodeMixRGB')
                mix_node.blend_type = 'MIX'
                mix_node.location = nloc_x + 300, nloc_y
                # add viewer node
                viewer_node = tree.nodes.new('CompositorNodeViewer')
                viewer_node.location = nloc_x + 500, nloc_y - 200
                links.new(mix_node.outputs['Image'], outfile_node.inputs['Image'])
                links.new(mix_node.outputs['Image'], viewer_node.inputs['Image'])
                # add background image node
                bg_image_node = tree.nodes.new('CompositorNodeImage')
                camera = find_object(camera_name)
                if camera is None:
                    print(f"Camera {camera_name} not found")
                    return
                bg_image_node.image = camera.data.background_images[0].image
                bg_image_node.location = nloc_x + 100, nloc_y + 300
                # add links
                links.new(bg_image_node.outputs['Image'], mix_node.inputs[1])
                links.new(rlayers_node.outputs[l.rendered_name], mix_node.inputs[2])
                # links.new(rlayers_node.outputs[C.mask.rendered_name], mix_node.inputs[0])
                links.new(full_mask_node.outputs['Value'], mix_node.inputs["Fac"])
                # add background output node
                outbg_node = add_output_node(tree, "background output",
                                             os.path.join(out_pass_dir, "background"),
                                             'PNG', '8', (nloc_x + 300, nloc_y + 300))
                out_paths["background"] = outbg_node.base_path
                links.new(bg_image_node.outputs['Image'], outbg_node.inputs['Image'])

    # Output combined shading
    loc_x += 200
    if C.diffuse_direct_shading in out_layers and C.diffuse_indirect_shading in out_layers:
        # add mix node
        mix_node = tree.nodes.new('CompositorNodeMixRGB')
        mix_node.blend_type = 'ADD'
        mix_node.location = loc_x, -400
        links.new(rlayers_node.outputs[C.diffuse_direct_shading.rendered_name], mix_node.inputs[1])
        links.new(rlayers_node.outputs[C.diffuse_indirect_shading.rendered_name], mix_node.inputs[2])
        # add denoise node
        loc_x += 200
        denoise_node = tree.nodes.new('CompositorNodeDenoise')
        denoise_node.prefilter = "ACCURATE"
        denoise_node.location = loc_x, -400
        links.new(mix_node.outputs['Image'], denoise_node.inputs['Image'])
        # add output node
        loc_x += 200
        outfile_node = tree.nodes.new('CompositorNodeOutputFile')
        outfile_node.base_path = os.path.join(out_pass_dir, "diffuse_shading")
        outfile_node.format.file_format = 'OPEN_EXR'
        outfile_node.format.color_depth = '32'
        outfile_node.location = loc_x, -400
        out_paths["diffuse_shading"] = outfile_node.base_path
        links.new(denoise_node.outputs['Image'], outfile_node.inputs['Image'])
    return out_paths


def configure_composite_AR_shadow_catcher(foreground_scene_name, output_path, camera_name, with_background):
    """
    Configure the composite nodes for rendering when shadows are rendered using shadow catcher.
    :param foreground_scene_name: the scene for rendering the foreground object
    :param output_path:
    :param camera_name:
    :param with_background:
    :return:
    """
    assert find_scene(foreground_scene_name) is not None, f"Scene {foreground_scene_name} not found"
    # Set up render passes
    activate_scene(foreground_scene_name)
    out_layers = [
        C.composite_image, C.diffuse_albedo,
        C.diffuse_direct_shading, C.diffuse_indirect_shading,
        C.normal, C.mask]
    setup_render_passes(clear_all=True, set_active_layers=out_layers)
    out_paths = {}

    # Enable compositing and clear default nodes
    clear_composite_nodes()
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Create rendered layer node
    rlayers_node = tree.nodes.new('CompositorNodeRLayers')
    rlayers_node.location = 0, 0
    rlayers_node.scene = find_scene(foreground_scene_name)

    # Create output file node
    out_pass_dir = os.path.join(output_path, "passes")
    loc_x, loc_y = 400, -200
    for l in out_layers:
        outfile_node = tree.nodes.new('CompositorNodeOutputFile')
        outfile_node.label = f"{l.rendered_name} output"
        outfile_node.base_path = os.path.join(out_pass_dir, l.pass_name)
        outfile_node.format.file_format = 'OPEN_EXR'
        outfile_node.format.color_depth = '32'
        outfile_node.location = loc_x, loc_y
        out_paths[l.pass_name] = outfile_node.base_path
        loc_y += 200
        links.new(rlayers_node.outputs[l.rendered_name], outfile_node.inputs['Image'])
        if l == C.composite_image:
            # add background
            if with_background:
                nloc_x, nloc_y = loc_x+100, loc_y
                # set output node
                outfile_node = tree.nodes.new('CompositorNodeOutputFile')
                outfile_node.label = "final image output"
                outfile_node.base_path = os.path.join(out_pass_dir, "final_image")
                outfile_node.format.file_format = 'PNG'
                outfile_node.format.color_depth = '8'
                outfile_node.location = nloc_x + 500, nloc_y
                out_paths["final_image"] = outfile_node.base_path
                # add mix node
                mix_node = tree.nodes.new('CompositorNodeMixRGB')
                mix_node.blend_type = 'MIX'
                mix_node.location = nloc_x + 300, nloc_y
                # add viewer node
                viewer_node = tree.nodes.new('CompositorNodeViewer')
                viewer_node.location = nloc_x + 500, nloc_y - 200
                links.new(mix_node.outputs['Image'], outfile_node.inputs['Image'])
                links.new(mix_node.outputs['Image'], viewer_node.inputs['Image'])
                # add background image node
                bg_image_node = tree.nodes.new('CompositorNodeImage')
                camera = find_object(camera_name)
                if camera is None:
                    print(f"Camera {camera_name} not found")
                    return
                bg_image_node.image = camera.data.background_images[0].image
                bg_image_node.location = nloc_x + 100, nloc_y + 300
                # add links
                links.new(rlayers_node.outputs[C.mask.rendered_name], mix_node.inputs[0])
                links.new(bg_image_node.outputs['Image'], mix_node.inputs[1])
                links.new(rlayers_node.outputs[l.rendered_name], mix_node.inputs[2])
                # add background output node
                outbg_node = add_output_node(tree, "background output",
                                             os.path.join(out_pass_dir, "background"),
                                             'PNG', '8', (nloc_x + 300, nloc_y + 300))
                out_paths["background"] = outbg_node.base_path
                links.new(bg_image_node.outputs['Image'], outbg_node.inputs['Image'])

    # Output combined shading
    loc_x += 200
    if C.diffuse_direct_shading in out_layers and C.diffuse_indirect_shading in out_layers:
        # add mix node
        mix_node = tree.nodes.new('CompositorNodeMixRGB')
        mix_node.blend_type = 'ADD'
        mix_node.location = loc_x, -400
        links.new(rlayers_node.outputs[C.diffuse_direct_shading.rendered_name], mix_node.inputs[1])
        links.new(rlayers_node.outputs[C.diffuse_indirect_shading.rendered_name], mix_node.inputs[2])
        # add denoise node
        loc_x += 200
        denoise_node = tree.nodes.new('CompositorNodeDenoise')
        denoise_node.prefilter = "ACCURATE"
        denoise_node.location = loc_x, -400
        links.new(mix_node.outputs['Image'], denoise_node.inputs['Image'])
        # add output node
        loc_x += 200
        outfile_node = tree.nodes.new('CompositorNodeOutputFile')
        outfile_node.base_path = os.path.join(out_pass_dir, "diffuse_shading")
        outfile_node.format.file_format = 'OPEN_EXR'
        outfile_node.format.color_depth = '32'
        outfile_node.location = loc_x, -400
        out_paths["diffuse_shading"] = outfile_node.base_path
        links.new(denoise_node.outputs['Image'], outfile_node.inputs['Image'])
    return out_paths


def get_transform(obect_name: str):
    obj = find_object(obect_name)
    if obj is None:
        raise ValueError(f"Object {obect_name} not found")
    # location
    loc = obj.location
    loc = np.array(loc, dtype=np.float32)
    # rotation
    if obj.rotation_mode != 'XYZ':
        raise ValueError(f"Rotation mode {obj.rotation_mode} not supported")
    rot = obj.rotation_euler  # (radians, rotation_mode)
    rot = np.array(rot, dtype=np.float32)
    # scale
    scale = obj.scale
    scale = np.array(scale, dtype=np.float32)
    return loc, rot, scale


def get_dimensions(obj_name: str):
    obj = find_object(obj_name)
    if obj is None:
        print(f"Object {obj_name} not found")
        return None
    dims = np.array(obj.dimensions, dtype=np.float32)
    return dims


def get_camera_parameters(camera_name: str) -> Optional[CameraParams]:
    camera = find_object(camera_name)
    if camera is None:
        print(f"Camera {camera_name} not found")
        return None

    # camera type
    cam_type = camera.data.type
    if cam_type != "PERSP":
        print(f"Camera type {cam_type} not supported")
        return None

    # camera FOV
    fov = camera.data.angle  # radians
    fov_axis = camera.data.sensor_fit  # HORIZONTAL, VERTICAL, AUTO

    # sensor size
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height

    # Get the render resolution
    render = bpy.context.scene.render
    resolution_x = render.resolution_x
    resolution_y = render.resolution_y
    scale = render.resolution_percentage / 100
    resolution_x *= scale
    resolution_y *= scale

    # camera location and rotation
    loc, rot, _ = get_transform(camera_name)

    return CameraParams(type=cam_type, fov=fov, fov_axis=fov_axis,
                        loc=loc, rot=rot,
                        img_width=int(resolution_x), img_height=int(resolution_y),
                        sensor_width=sensor_width, sensor_height=sensor_height)


def enable_addon(addon_module_name):
    bpy.ops.preferences.addon_enable(module=addon_module_name)


def disable_addon(addon_module_name):
    bpy.ops.preferences.addon_disable(module=addon_module_name)


def use_sky_texture_for_world_surface(clear_world_node=True, strength=1.0,
                                      sky_params=None):
    """
    Use NISHITA sky texture for world surface.
    :param clear_world_node:
    :param strength:
    :param sky_params:
    :return:
    """

    bpy.context.scene.world.use_nodes = True
    if clear_world_node:
        bpy.context.scene.world.node_tree.nodes.clear()
    sky_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeTexSky')
    sky_node.sky_type = 'NISHITA'
    sun_size_deg = 1
    air_density, dust_density, ozone_density = 1.0, 1.0, 1.0
    if sky_params is not None:
        if isinstance(sky_params, dict):
            sun_size_deg = sky_params.get("sun_size_deg", 1)
            air_density = sky_params.get("air_density", 1.0)
            dust_density = sky_params.get("dust_density", 1.0)
            ozone_density = sky_params.get("ozone_density", 1.0)
        else:
            print(f"Warning: sun_params should be a dictionary, but got {type(sky_params)}. Using default values.")
    sky_node.sun_size = np.deg2rad(sun_size_deg)
    sky_node.air_density = air_density
    sky_node.dust_density = dust_density
    sky_node.ozone_density = ozone_density
    sky_node.location = 0, 0
    bg_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = strength
    bg_node.location = 200, 0
    output_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeOutputWorld')
    output_node.location = 400, 0
    bpy.context.scene.world.node_tree.links.new(sky_node.outputs['Color'], bg_node.inputs['Color'])
    bpy.context.scene.world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])


def set_world_surface(environment_lighting_path=None, strength=1.0):
    bpy.context.scene.world.use_nodes = True
    bpy.context.scene.world.node_tree.nodes.clear()
    if environment_lighting_path is None:
        # add output node
        output_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeOutputWorld')
        return
    # add environment texture node
    env_tex_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    env_tex_node.image = bpy.data.images.load(environment_lighting_path)
    env_tex_node.image.source = 'FILE'
    env_tex_node.image.colorspace_settings.name = 'Linear'
    env_tex_node.image.alpha_mode = 'PREMUL'
    env_tex_node.interpolation = 'Linear'
    env_tex_node.projection = 'EQUIRECTANGULAR'
    env_tex_node.location = 0, 0
    if environment_lighting_path.split('.')[-1] != 'exr':
        print(f"Warning: Environment lighting image format is not EXR, thus it may not be in linear space.")
    # add background node
    bg_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = strength
    bg_node.location = 300, 0
    # add output node
    output_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeOutputWorld')
    output_node.location = 500, 0
    # link nodes
    bpy.context.scene.world.node_tree.links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])
    bpy.context.scene.world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    # Rotate the environment map
    # By observation, the image is mapped as (https://docs.blender.org/manual/en/latest/render/shader_nodes/textures/environment.html)
    # P_required = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 1]], dtype=np.float32)
    # P_tex_coord = np.array([
    #     [0, 0, -1],
    #     [-1, 0, 0],
    #     [0, 1, 0]], dtype=np.float32)
    # P_required = R @ P_tex_coord
    # By computation, R is the rotation with angles [-90, 90, 0] degrees

    # texture coordinate node
    tex_coord_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeTexCoord')
    tex_coord_node.location = -400, 0
    # mapping node
    mapping_node = bpy.context.scene.world.node_tree.nodes.new('ShaderNodeMapping')
    mapping_node.vector_type = 'TEXTURE'
    rot_deg = np.array([-90, 90, 0], dtype=np.float32)
    mapping_node.inputs["Rotation"].default_value = np.deg2rad(rot_deg)
    mapping_node.location = -200, 0
    # link nodes
    bpy.context.scene.world.node_tree.links.new(tex_coord_node.outputs['Generated'], mapping_node.inputs['Vector'])
    bpy.context.scene.world.node_tree.links.new(mapping_node.outputs['Vector'], env_tex_node.inputs['Vector'])


def is_sun_object(selected_object):
    # Check if the selected object is a Sun light
    if selected_object and selected_object.type == 'LIGHT' and selected_object.data.type == 'SUN':
        return True
    else:
        print(f"The selected object ({getattr(selected_object, 'name', None)}) is not a Sun light.")
        return False


def get_light_info(light_name):
    # Check if the object exists in the scene
    obj = find_object(light_name)
    if obj is None:
        print(f"Object {light_name} not found.")
        return None

    # Check if the object is a light
    if obj.type != 'LIGHT':
        print(f"Object '{light_name}' is not a light source.")
        return None

    light_data = obj.data

    # Get light type
    light_type = light_data.type

    # Get light parameters based on type
    if light_type == 'POINT':
        return {
            "type": light_type,
            "color": list(light_data.color),
            "energy": light_data.energy,
            "shadow_soft_size": light_data.shadow_soft_size,
        }
    elif light_type == 'SUN':
        return {
            "type": light_type
        }
    elif light_type == 'SPOT':
        return {
            "type": light_type,
            "color": list(light_data.color),
            "energy": light_data.energy,
            "shadow_soft_size": light_data.shadow_soft_size,
            "spot_size": light_data.spot_size,
            "spot_blend": light_data.spot_blend
        }
    elif light_type == 'AREA':
        raise NotImplementedError("Area light information is not implemented.")
    else:
        print(f"Error: unknown light type: {light_type}")
        return None


def add_sun_and_sky_texture(clear_world_node=True,
                            sky_strength=1.0,
                            sun_name="Sun_main",
                            sun_params=None,
                            sky_params=None,
                            remove_existing_sun=False):
    """
    Use "sun_position" addon to add a sky texture (with sun) to the world surface.
    :param clear_world_node:
    :param sky_strength:
    :param sun_name:
    :param sun_params: parameters for "sun_position" addon to set the sun direction
    :param sky_params: parameters for "NISHITA" sky texture to set the sky appearance
    :param remove_existing_sun:
    :return:
    """

    # Enable addon
    addon_module_name = "sun_position"
    enable_addon(addon_module_name)

    # Use sky texture for world surface
    use_sky_texture_for_world_surface(clear_world_node=clear_world_node,
                                      strength=sky_strength,
                                      sky_params=sky_params)

    # Add sun: sun is used to get the sun direction
    if find_object(sun_name):
        print(f"Sun object {sun_name} already exists.")
        if remove_existing_sun:
            bpy.data.objects.remove(find_object(sun_name))
            print(f"Removed existing sun object {sun_name}.")
        else:
            return
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
    bpy.ops.object.light_add(type='SUN', align='WORLD', location=(0, 0, 0))
    sun_object = bpy.context.object
    if not is_sun_object(sun_object):
        print("The selected object is not a Sun light.")
        return
    sun_object.name = sun_name
    sun_object.data.energy = 0
    set_object_visibility(sun_name, hide_render=True, hide_viewport=True)
    print(f"Sun object {sun_name} added.")

    # Link sun object and sky texture to the sun_position addon which automatically updates these two items
    bpy.context.scene.sun_pos_properties.sun_object = sun_object
    bpy.context.scene.sun_pos_properties.sky_texture = "Sky Texture"
    if sun_params is None:
        warnings.warn("Sun parameters are not provided. Using default values.")
        return
    for k, v in sun_params.items():
        if not hasattr(bpy.context.scene.sun_pos_properties, k):
            print(f"Attribute {k} not found in sun_pos_properties")
            continue
        setattr(bpy.context.scene.sun_pos_properties, k, v)


def set_object_visibility(obj_name, hide_render=True, hide_viewport=True):
    obj = find_object(obj_name)
    if obj is None:
        print(f"Object {obj_name} not found.")
        return
    obj.hide_render = hide_render
    # check if the object is in the view layer
    if obj_name in bpy.context.view_layer.objects:
        obj.hide_set(hide_viewport)


def set_collection_visibility(collection_name, hide_render=True, hide_viewport=True):
    collection = find_collection(collection_name)
    if collection:
        collection.hide_render = hide_render
        collection.hide_viewport = hide_viewport
    else:
        print(f"Collection '{collection_name}' does not exist")


def set_object_transform_and_scale(obj_name, loc=None, rot=None, scale=None):
    """
    Set the location, rotation, and scale of the object.
    :param obj_name:
    :param loc:
    :param rot: radians
    :param scale:
    :return:
    """
    obj = find_object(obj_name)
    if obj is None:
        print(f"Object {obj_name} not found.")
        return
    if loc is not None:
        obj.location = loc
    if rot is not None:
        obj.rotation_euler = rot
    if scale is not None:
        obj.scale = scale


def move_object_to_collection(obj_name, collection_name):
    # Move the object to the collection
    obj = find_object(obj_name)
    if obj is None:
        print(f"Object {obj_name} not found.")
        return
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    if obj.name not in collection.objects:
        collection.objects.link(obj)
    # Unlink the original collection
    for coll in obj.users_collection:
        if coll.name != collection_name:
            coll.objects.unlink(obj)


def deactivate_collection_object(collection_name):
    # Deactivate all objects in the collection
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        print(f"Collection {collection_name} not found.")
        return
    for obj in collection.objects:
        set_object_visibility(obj.name, hide_render=True, hide_viewport=True)
    print(f"Collection {collection_name} deactivated.")


def remove_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        print(f"Collection {collection_name} not found.")
        return
    bpy.data.collections.remove(collection)
    print(f"Collection {collection_name} removed.")


def setup_environment_camera_for_render(main_camera_name, obj_name,
                                        loc_x=None, loc_y=None, loc_z=None,
                                        loc_shift=None):
    """
    Setup an environment camera for rendering the environment map.
    The camera is placed at the object location and has the same rotation as the main camera.
    :param main_camera_name:
    :param obj_name:
    :param loc_shift:
    :return:
    """
    # Put the camera at the object location and set the same rotation as the main camera
    object = find_object(obj_name)
    if object is None:
        print(f"Object {obj_name} not found.")
        return
    loc, _, _ = get_transform(obj_name)
    if loc_x is not None:
        loc[0] = loc_x
    if loc_y is not None:
        loc[1] = loc_y
    if loc_z is not None:
        loc[2] = loc_z
    if loc_shift is not None:  # shift the location
        loc = np.array(loc, dtype=np.float32) + np.array(loc_shift, dtype=np.float32)
    _, rot, _ = get_transform(main_camera_name)
    rot = np.rad2deg(rot)
    if len(obj_name) > 10:
        env_camera_name = f"{obj_name[:10]}_env"
    else:
        env_camera_name = f"{obj_name}_env"
    setup_render_camera(env_camera_name, type="PANO", fov_axis="HORIZONTAL",
                        loc=loc, rot_deg=rot)  # set up the panorama camera
    return env_camera_name


def setup_background_image(camera_name, img_path):
    """
    Set background image for the camera view. Note that this not influence the rendering.
    :param camera_name:
    :param img_path:
    :return:
    """

    # Get camera object
    camera = find_object(camera_name)
    if camera is None:
        print(f"Camera {camera_name} not found.")
        return
    # Set background image
    camera.data.show_background_images = True

    # Clear existing background images
    camera.data.background_images.clear()

    # Add new background image
    bg_img = camera.data.background_images.new()
    bg_img.image = bpy.data.images.load(img_path)
    bg_img.image.source = 'FILE'
    bg_img.image.colorspace_settings.name = 'sRGB'
    bg_img.display_depth = 'BACK'
    bg_img.alpha = 1.0
    bg_img.frame_method = 'FIT'


def setup_image_texture_node(tex_image_node, img_path,
                             image_source='FILE', colorspace='Linear', alpha_mode='PREMUL',
                             interpolation='Cubic', projection='FLAT', extension='CLIP'):
    tex_image_node.image = bpy.data.images.load(img_path)
    tex_image_node.image.source = image_source
    tex_image_node.image.colorspace_settings.name = colorspace
    tex_image_node.image.alpha_mode = alpha_mode
    tex_image_node.interpolation = interpolation
    tex_image_node.projection = projection
    tex_image_node.extension = extension


def add_sun_light(light_name, loc, rot, energy,
                  max_bounces=1024, remove_existing=False):
    """
    Add a Sun light.
    :param light_name:
    :param loc:
    :param rot: XYZ Euler angles in radians
    :param energy:
    :param remove_existing:
    :return:
    """
    if find_object(light_name) and remove_existing:
        bpy.data.objects.remove(find_object(light_name))
        print(f"Removed existing sun light object: {light_name}.")
    light = find_object(light_name)
    if light is None:
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
        bpy.ops.object.light_add(type='SUN', align='WORLD', location=loc, rotation=rot)
        light = bpy.context.object
    assert light is not None, f"Sun light object {light_name} not found."
    assert is_sun_object(light), f"Object {light_name} is not a Sun light."
    light.name = light_name
    set_object_transform_and_scale(light_name, loc=loc, rot=rot, scale=None)
    light.data.energy = energy
    light.data.cycles.max_bounces = max_bounces
    set_object_visibility(light_name, hide_render=False, hide_viewport=False)
    return light


def add_area_light(light_name, loc, rot, width, height, power,
                   spread_deg,
                   image_path,
                   use_multiple_importance_sampling=False,
                   remove_existing=False):
    """
    Add an area light with an image texture.
    :param light_name:
    :param loc:
    :param rot: XYZ Euler angles in radians
    :param width:
    :param height:
    :param power:
    :param spread_deg: Light spread angle in degrees
    :param image_path:
    :param use_multiple_importance_sampling:
    :param remove_existing:
    :return:
    """
    if find_object(light_name):
        if remove_existing:
            bpy.data.objects.remove(find_object(light_name))
            print(f"Area light {light_name} removed.")
        else:
            print(f"Area light {light_name} already exists.")
            return
    set_object_visibility(light_name, hide_render=False, hide_viewport=False)
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
    bpy.ops.object.light_add(type='AREA', align='WORLD', location=loc, rotation=rot)
    light = bpy.context.object
    light.name = light_name
    light.data.energy = power
    light.data.shape = 'RECTANGLE'
    light.data.size = width
    light.data.size_y = height
    # light.cycles.max_bounces = 32
    light.data.cycles.cast_shadow = True
    light.data.cycles.use_multiple_importance_sampling = use_multiple_importance_sampling
    if "3.6." in blender_version():
        print(f"Blender 3.6 has a bug in multiple importance sampling for area light."
              f"Thus multiple importance sampling is disabled automatically. See more details at "
              f"https://projects.blender.org/blender/blender/issues/123448.")
        light.data.cycles.use_multiple_importance_sampling = False
    light.data.spread = np.deg2rad(spread_deg)

    set_object_visibility(light_name, hide_render=False, hide_viewport=False)

    # Set the image texture
    light.data.use_nodes = True
    light.data.node_tree.nodes.clear()
    tex_image = light.data.node_tree.nodes.new('ShaderNodeTexImage')
    setup_image_texture_node(tex_image, image_path)
    tex_image.location = 0, 0

    # Add geometry node, emission node, and output node
    geo_node = light.data.node_tree.nodes.new('ShaderNodeNewGeometry')
    geo_node.location = -200, 0
    emission_node = light.data.node_tree.nodes.new('ShaderNodeEmission')
    emission_node.inputs['Strength'].default_value = 1.0
    emission_node.location = 300, 0
    output_node = light.data.node_tree.nodes.new('ShaderNodeOutputLight')
    output_node.location = 500, 0

    # Link the nodes
    light.data.node_tree.links.new(geo_node.outputs['Parametric'], tex_image.inputs['Vector'])
    light.data.node_tree.links.new(tex_image.outputs['Color'], emission_node.inputs['Color'])
    light.data.node_tree.links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
    return light


def set_material_to_object(obj_name, mat_name, invisible_reflection_and_transmission,
                           albedo_map_path=None,
                           albedo_mask_path=None):
    obj = find_object(obj_name)
    if obj is None:
        print(f"Object {obj_name} not found.")
        return
    # find existing material or create a new one
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
    # set the material to the object
    if not obj.data.materials:
        obj.data.materials.append(mat)
    else:
        # Assign the material to the first material slot
        obj.data.materials[0] = mat
    # configure the material
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links
    loc_x, loc_y = 0, 0
    if invisible_reflection_and_transmission and albedo_map_path is not None:
        print(f"Error: transparent plane cannot have albedo map.")
        return
    if invisible_reflection_and_transmission:
        #  Add material output
        output_node = tree.nodes.new('ShaderNodeOutputMaterial')
        output_node.location = loc_x, loc_y
        loc_x -= 200
        # Add mix shader
        mix_node = tree.nodes.new('ShaderNodeMixShader')
        mix_node.location = loc_x, loc_y
        loc_x -= 200
        # Add addition node
        math_node_greater = tree.nodes.new('ShaderNodeMath')
        math_node_greater.operation = 'GREATER_THAN'
        math_node_greater.inputs[1].default_value = 0.5
        math_node_greater.location = loc_x, loc_y
        loc_y -= 200
        # Add diffuse node
        diffuse_node = tree.nodes.new('ShaderNodeBsdfDiffuse')
        diffuse_node.inputs['Roughness'].default_value = 0.0
        diffuse_node.location = loc_x, loc_y
        loc_y -= 200
        # Add transparent node
        transparent_node = tree.nodes.new('ShaderNodeBsdfTransparent')
        transparent_node.location = loc_x, loc_y
        loc_x -= 200
        loc_y += 400
        # Add greater than node
        math_node_add = tree.nodes.new('ShaderNodeMath')
        math_node_add.operation = 'ADD'
        math_node_add.location = loc_x, loc_y
        loc_x -= 200
        # Add light path node
        light_path_node = tree.nodes.new('ShaderNodeLightPath')
        light_path_node.location = loc_x, loc_y
        # Add links
        links.new(light_path_node.outputs['Is Reflection Ray'], math_node_add.inputs[0])
        links.new(light_path_node.outputs['Is Transmission Ray'], math_node_add.inputs[1])
        links.new(math_node_add.outputs['Value'], math_node_greater.inputs[0])
        links.new(math_node_greater.outputs['Value'], mix_node.inputs["Fac"])
        links.new(diffuse_node.outputs['BSDF'], mix_node.inputs[1])
        links.new(transparent_node.outputs['BSDF'], mix_node.inputs[2])
        links.new(mix_node.outputs['Shader'], output_node.inputs['Surface'])
    elif albedo_map_path is not None:
        if albedo_mask_path is None:
            print(f"Error: albedo mask path is not provided.")
            return
        loc_x -= 700
        loc_y += 300
        # Add texture coordinate node
        texture_coordinate_node = nodes.new(type='ShaderNodeTexCoord')
        texture_coordinate_node.location = loc_x, loc_y
        loc_x += 200
        loc_y += 100
        # Add image texture node for segmentation map
        seg_map_node = nodes.new(type='ShaderNodeTexImage')
        setup_image_texture_node(seg_map_node, albedo_mask_path,
                                 alpha_mode="STRAIGHT")
        seg_map_node.location = loc_x, loc_y
        loc_y -= 400
        # Add image texture node for albedo map
        albedo_node = nodes.new(type='ShaderNodeTexImage')
        setup_image_texture_node(albedo_node, albedo_map_path)
        albedo_node.location = loc_x, loc_y
        loc_x += 300
        loc_y += 200
        # Add transparent node
        transparent_bsdf_node = nodes.new(type='ShaderNodeBsdfTransparent')
        transparent_bsdf_node.location = loc_x, loc_y
        loc_y -= 100
        # Add Lambertian node
        diffuse_bsdf_node = nodes.new(type='ShaderNodeBsdfDiffuse')
        diffuse_bsdf_node.location = loc_x, loc_y
        loc_x += 200
        loc_y += 200
        # Add mix shader node
        mix_shader_node = nodes.new(type='ShaderNodeMixShader')
        mix_shader_node.location = loc_x, loc_y
        loc_x += 200
        # Add material output node
        material_output_node = nodes.new(type='ShaderNodeOutputMaterial')
        material_output_node.location = loc_x, loc_y
        # Link nodes
        links.new(texture_coordinate_node.outputs['Window'], albedo_node.inputs['Vector'])
        links.new(texture_coordinate_node.outputs['Window'], seg_map_node.inputs['Vector'])
        links.new(albedo_node.outputs['Color'], diffuse_bsdf_node.inputs['Color'])
        links.new(seg_map_node.outputs['Color'], mix_shader_node.inputs['Fac'])
        links.new(transparent_bsdf_node.outputs['BSDF'], mix_shader_node.inputs[1])
        links.new(diffuse_bsdf_node.outputs['BSDF'], mix_shader_node.inputs[2])
        links.new(mix_shader_node.outputs['Shader'], material_output_node.inputs['Surface'])


def cancel_non_Lambertian_effects(obj_name):
    """
    Remove some non-Lambertian effects from the material of the object.
    :param obj_name:
    :return:
    """
    # Find the object
    obj = find_object(obj_name)
    if obj is None:
        print(f"Object {obj_name} not found.")
        return

    # Check if the object has materials
    if not obj.data.materials:
        print(f"Object {obj_name} has no materials.")
        return

    mat = obj.data.materials[0]
    mat.use_nodes = True

    # Check if the material node tree has nodes
    if not mat.node_tree.nodes:
        print(f"Material {mat.name} has no nodes.")
        return

    # Find the Principled BSDF node in the node tree
    bsdf_node = None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            bsdf_node = node
            # Not use subsurface scattering
            if bsdf_node.inputs['Subsurface'].is_linked:
                for link in bsdf_node.inputs['Subsurface'].links:
                    mat.node_tree.links.remove(link)
            bsdf_node.inputs['Subsurface'].default_value = 0
            # Cancel metallic effect
            if bsdf_node.inputs['Metallic'].is_linked:
                for link in bsdf_node.inputs['Metallic'].links:
                    mat.node_tree.links.remove(link)
            bsdf_node.inputs['Metallic'].default_value = 0
            # Cancel specular effects
            if bsdf_node.inputs['Specular'].is_linked:
                for link in bsdf_node.inputs['Specular'].links:
                    mat.node_tree.links.remove(link)
            bsdf_node.inputs['Specular'].default_value = 0
            # IOR
            if bsdf_node.inputs['IOR'].is_linked:
                for link in bsdf_node.inputs['IOR'].links:
                    mat.node_tree.links.remove(link)
            bsdf_node.inputs['IOR'].default_value = 1.0
            # Cancel transmission effect
            if bsdf_node.inputs['Transmission'].is_linked:
                for link in bsdf_node.inputs['Transmission'].links:
                    mat.node_tree.links.remove(link)
            bsdf_node.inputs['Transmission'].default_value = 0
            # Set Emission Strength to 0
            if bsdf_node.inputs['Emission Strength'].is_linked:
                for link in bsdf_node.inputs['Emission Strength'].links:
                    mat.node_tree.links.remove(link)
            bsdf_node.inputs['Emission Strength'].default_value = 0

    if bsdf_node is None:
        print(f"Material {mat.name} does not contain a Principled BSDF node.")
        return


def add_plane(plane_name, loc, rot, scale, shadow_catcher,
              albedo_map_path=None,
              albedo_mask_path=None,
              transparent=False, remove_existing=False):
    if find_object(plane_name):
        if remove_existing:
            bpy.data.objects.remove(find_object(plane_name))
            print(f"Plane {plane_name} removed.")
        else:
            print(f"Plane {plane_name} already exists.")
            return
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection
    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=loc, rotation=rot, scale=scale)
    plane = bpy.context.object
    plane.name = plane_name
    plane.is_shadow_catcher = shadow_catcher
    plane.scale = scale
    if transparent and albedo_map_path is not None:
        print(f"Error: transparent plane cannot have albedo map.")
        return
    if transparent:
        set_material_to_object(plane_name, f"{plane_name}_transparent", invisible_reflection_and_transmission=True)
    if albedo_map_path is not None:
        set_material_to_object(plane_name, f"{plane_name}_textured",
                               invisible_reflection_and_transmission=False,
                               albedo_map_path=albedo_map_path,
                               albedo_mask_path=albedo_mask_path)
    set_object_visibility(plane_name, hide_render=False, hide_viewport=False)
    return plane


def get_location_and_normal_at_cursor():
    # This function was largely contributed by ChatGPT-4o. May have bugs.

    # Get the active object
    obj = bpy.context.active_object

    if obj is None:
        print("No active object selected.")
        return None

    # Get the 3D cursor location
    cursor_location = bpy.context.scene.cursor.location

    # Ensure we are in object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create a BMesh from the object
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Transform cursor location to object's local space
    cursor_location_local = obj.matrix_world.inverted() @ cursor_location

    # Find the nearest face to the cursor location
    closest_face = None
    closest_distance = float('inf')
    for face in bm.faces:
        # Calculate the distance from the cursor to the face center
        face_center = face.calc_center_median()
        distance = (face_center - cursor_location_local).length
        if distance < closest_distance:
            closest_face = face
            closest_distance = distance

    if closest_face is None:
        print("No face found near the cursor location.")
        bm.free()
        return None

    # Get the normal of the closest face and transform it to world space
    normal = closest_face.normal
    normal_world = obj.matrix_world.to_3x3() @ normal
    bm.free()

    # The normal should be pointing towards the camera
    camera = bpy.context.scene.camera
    camera_location = camera.location
    camera_direction = camera_location - cursor_location
    if normal_world @ camera_direction < 0:
        print("Flipping the normal direction.")
        normal_world = -normal_world
    return{
        "location": cursor_location,
        "normal": normal_world
    }


def get_closest_point_on_surface(surface_name, location):
    # This function was largely contributed by ChatGPT-4o. May have bugs.
    """
    Get the closest point on the surface of the object to the given location.
    :param surface_name:
    :param location:
    :return:
    """

    # Get the object by name
    obj = find_object(surface_name)
    if obj is None:
        print(f"Object named '{surface_name}' not found.")
        return None

    #     # Ensure we are in object mode
    # bpy.ops.object.mode_set(mode='OBJECT')


    # Create a BMesh from the object
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Transform location to object's local space
    location = mathutils.Vector(location)
    location_local = obj.matrix_world.inverted() @ location

    # Find the nearest point on the surface
    closest_point = None
    closest_distance = float('inf')
    for face in bm.faces:
        # Get the vertices of the face
        verts = face.verts
        if len(verts) == 3:
            # Triangle face
            p1, p2, p3 = [v.co for v in verts]
            closest_point_on_face = mathutils.geometry.closest_point_on_tri(location_local, p1, p2, p3)
        elif len(verts) == 4:
            # Quad face, split into two triangles
            p1, p2, p3, p4 = [v.co for v in verts]
            closest_point_on_face_1 = mathutils.geometry.closest_point_on_tri(location_local, p1, p2, p3)
            closest_point_on_face_2 = mathutils.geometry.closest_point_on_tri(location_local, p1, p3, p4)
            # Choose the closest of the two
            dist1 = (closest_point_on_face_1 - location_local).length
            dist2 = (closest_point_on_face_2 - location_local).length
            if dist1 < dist2:
                closest_point_on_face = closest_point_on_face_1
            else:
                closest_point_on_face = closest_point_on_face_2
        else:
            print(f"Face with {len(verts)} vertices is not supported in closest point calculation, "
                  f" and is skipped.")

        distance = (closest_point_on_face - location_local).length
        if distance < closest_distance:
            closest_distance = distance
            closest_point = closest_point_on_face

    bm.free()

    if closest_point is None:
        print("No closest point found.")
        return None

    # Transform the closest point back to world space
    closest_point_world = obj.matrix_world @ closest_point
    closest_length = (closest_point_world - location).length
    closest_point_world = np.array(closest_point_world, dtype=np.float32)
    return closest_point_world, closest_length