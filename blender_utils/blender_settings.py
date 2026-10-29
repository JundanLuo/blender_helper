import sys
import os
import importlib
import json


from blender_helper.blender_utils import blender_operation as bop
from blender_helper.blender_utils import constants as C
importlib.reload(bop)
importlib.reload(C)
print("Reload: bop, C")


def set_parameters(cfg, object_with_mask=None):
    # Set output directory
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("Output directory: ",  cfg["output_dir"])

    # Setup render camera
    set_persp_main_camera = "CAMERA_MAIN" in cfg.keys()
    if not set_persp_main_camera:
        print("CAMERA_MAIN not found in config file. Skip setting up the perspective camera.")
    else:
        loc = cfg["CAMERA_MAIN"]["location"] if cfg["CAMERA_MAIN"]["location"] != "" else None
        rot_deg = cfg["CAMERA_MAIN"]["rotation_deg"] if cfg["CAMERA_MAIN"]["rotation_deg"] != "" else None
        bop.setup_render_camera(
            camera_name=cfg["CAMERA_MAIN"]["name"],
            type="PERSP",
            fov_deg=cfg["CAMERA_MAIN"]["fov_deg"],
            fov_axis=cfg["CAMERA_MAIN"]["fov_axis"],
            clip=(cfg["CAMERA_MAIN"]["clip_start"], cfg["CAMERA_MAIN"]["clip_end"]),
            loc=loc,
            rot_deg=rot_deg
        )

    # Configure render engine
    bop.configure_render_engine(
        engine=cfg["RENDERER"]["engine"],
        sample_count=cfg["RENDERER"]["sample_count"],
        time_limit=cfg["RENDERER"]["time_limit"],
        max_bounces=cfg["RENDERER"]["max_bounces"],
        use_gpu=cfg["RENDERER"]["use_gpu"],
        transparent_bg=cfg["SCENE"]["transparent_bg"]
    )

    # Configure output
    bop.configure_output(
        output_path=cfg["output_dir"],
        update_camera_sensor_for_square_pixel=set_persp_main_camera,
        img_width=cfg["SCENE"]["img_width"],
        img_height=cfg["SCENE"]["img_height"],
        output_format=cfg["RENDERER"]["output_format"]
    )

    # Configure output passes layers
    out_passes_layers = [getattr(C, layer) for layer in cfg["RENDERER"]["out_passes_layers"]]

    # Setup composite nodes
    outfile_node_label_dict, out_paths = bop.configure_composite(cfg["output_dir"], out_passes_layers,
                                                                 object_with_mask=object_with_mask)
    return {"outfile_node_label_dict": outfile_node_label_dict,
            "out_paths": out_paths}


def save_camera_and_scene_information(cfg, obj_list=None):
    """
    Save camera and scene information to json files
    :param cfg:
    :param obj_list:
    :return:
    """

    # Save camera params
    camera_name = cfg["CAMERA_MAIN"]["name"]
    cam_params = bop.get_camera_parameters(camera_name)
    out_cam_file = os.path.join(cfg["output_dir"], "camera_params.json")
    with open(out_cam_file, 'w') as f:
        json.dump(cam_params.to_dict(), f, indent=4)

    # Scene information
    scene_inf = {}

    # Primary light params
    if "PRIMARY_LIGHT" in cfg.keys():
        pl_name = cfg["PRIMARY_LIGHT"]["name"]
        pl_obj = bop.find_object(pl_name)
        if pl_obj is not None:
            pl_loc, pl_rot, _ = bop.get_transform(pl_name)
            pl_params = {
                "name": pl_name,
                "location": pl_loc.tolist(),
                "rotation": pl_rot.tolist()
            }
            light_info = bop.get_light_info(pl_name)
            if light_info is None:
                assert False, f"Error: get light info of {pl_name} failed."
            else:
                common_keys = set(pl_params.keys()).intersection(light_info.keys())
                assert len(common_keys) == 0, \
                    f"Error: common keys {common_keys} found in pl_params and light_info."
                pl_params.update(light_info)
            scene_inf["primary_light_params"] = pl_params
        else:
            assert False, f"Primary light object {pl_name} not found."
    else:
        print(f"WARN: Primary light not found in config file.")

    # Object params
    if obj_list is not None:
        obj_params = {}
        for obj in obj_list:
            obj_name = obj["name"]
            obj_loc, obj_rot, obj_scale = bop.get_transform(obj_name)
            obj_dims = bop.get_dimensions(obj_name)
            visible = not bop.find_object(obj_name).hide_render
            obj_params[obj_name] = {
                "location": obj_loc.tolist(),
                "rotation": obj_rot.tolist(),
                "scale": obj_scale.tolist(),
                "dimensions": obj_dims.tolist(),
                "closest_surface_point": obj["closest_surface_point"],
                "closest_distance": obj["closest_distance"],
                "visible": visible,
            }
        scene_inf["object_params"] = obj_params

    # Save scene params
    out_scene_file = os.path.join(cfg["output_dir"], "scene_params.json")
    with open(out_scene_file, 'w') as f:
        json.dump(scene_inf, f, indent=4)
