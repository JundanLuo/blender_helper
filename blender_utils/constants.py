from collections import namedtuple
import math

LayerStrMapping = namedtuple('LayerStrMapping', ['pass_name', 'rendered_name'])

composite_image = LayerStrMapping("combined", "Image")
normal = LayerStrMapping("normal", "Normal")
depth = LayerStrMapping("z", "Depth")
diffuse_direct_shading = LayerStrMapping("diffuse_direct", "DiffDir")
diffuse_indirect_shading = LayerStrMapping("diffuse_indirect", "DiffInd")
diffuse_albedo = LayerStrMapping("diffuse_color", "DiffCol")
mask = LayerStrMapping("alpha", "Alpha")
object_mask = LayerStrMapping("cryptomatte_object", None)
shadow_catcher = LayerStrMapping("shadow_catcher", "Shadow Catcher")

GAMMA = 1.0/2.2

DEFAULT_SUN_DIRECTION = [0, 0, -1]
DEFAULT_LIGHT_DIRECTION = [0, 0, -1]

DEFAULT_SUN_PARAMS = {
    "latitude": 22.5,
    "longitude": -43.55,
    "show_north": False,
    "north_offset": -1.30725,
    "use_refraction": True,
    "sun_distance": 50,
    "use_day_of_year": False,
    "day": 8,
    "month": 1,
    "year": 2023,
    "time": 15,
    "UTC_zone": 0,
    "use_daylight_savings": False,
}

DEFAULT_SKY_PARAMS = {
    "sun_size_deg": 0.5,  # Note that zero-size sun is invisible in the sky, thus can not be captured by the camera
    "air_density": 1,
    "dust_density": 1,
    "ozone_density": 1,
    "world_strength": 1
}

# the ratio to convert from energy to radiance was wrong in 3.6: https://projects.blender.org/blender/blender/issues/123448
AREA_LIGHT_POWER_3_3 = math.pi
AREA_LIGHT_POWER_3_6 = 1.0 / 0.250
AREA_LIGHT_POWER_4_1 = math.pi