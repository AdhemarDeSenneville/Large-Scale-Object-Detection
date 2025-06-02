from functools import lru_cache

from pyproj import CRS

def format_crs(crs_input):

    crs_obj = CRS.from_user_input(crs_input)
    # If it has an authority code (e.g. EPSG:XXXX), return that
    auth = crs_obj.to_authority()
    if auth:
        return f"{auth[0]}:{auth[1]}"
    # Otherwise fall back to its proj4 or WKT string
    return crs_obj.to_string()

@lru_cache(maxsize=None)
def crs_units(crs_input):
    """
    Return:
      - 'angles' if the crs is geographic (degrees)
      - 'metre'  if projected in metres
      - ''       otherwise
    """
    crs_code = format_crs(crs_input)


    # ask pyproj for its first axis unit
    crs_obj = CRS.from_user_input(crs_code)
    unit = crs_obj.axis_info[0].unit_name.lower()

    if "degree" in unit:
        return "angles"
    if "metre" in unit:
        return "metre"