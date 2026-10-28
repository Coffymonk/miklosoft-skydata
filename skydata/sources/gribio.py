"""Read one field from a GRIB2 file on a regular latitude-longitude grid (with ecCodes)."""
import eccodes
import numpy as np


def read_latlon(fn, short_name=None):
    """values[lat, lon] with latitude rising with the row, and the grid description."""
    with open(fn, "rb") as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                raise ValueError(f"{short_name or 'field'} not found in {fn}")
            try:
                if short_name and eccodes.codes_get(h, "shortName") != short_name:
                    continue
                ni = eccodes.codes_get(h, "Ni")
                nj = eccodes.codes_get(h, "Nj")
                la1 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
                la2 = eccodes.codes_get(h, "latitudeOfLastGridPointInDegrees")
                lo1 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
                lo2 = eccodes.codes_get(h, "longitudeOfLastGridPointInDegrees")
                v = eccodes.codes_get_values(h).reshape(nj, ni).astype(np.float32)
                miss = eccodes.codes_get(h, "missingValue")
                if eccodes.codes_get(h, "bitmapPresent"):
                    v[v == miss] = np.nan
            finally:
                eccodes.codes_release(h)
            if la1 > la2:            # north to south: flip so rows go south to north
                v = v[::-1]
                la1, la2 = la2, la1
            if lo1 >= 180:
                lo1 -= 360
            if lo2 >= 180:
                lo2 -= 360
            dlat = (la2 - la1) / (nj - 1)
            dlon = ((lo2 - lo1) % 360) / (ni - 1) if ni > 1 else 1.0
            return v, dict(lat0=la1, lon0=lo1, dlat=dlat, dlon=dlon, global_=abs(dlon * ni - 360) < 1e-3)
