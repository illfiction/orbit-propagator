import numpy as np

class GroundStation():
    def __init__(self, name, lat_deg, lon_deg, alt_km,min_elevation_deg):
        self.name = name
        self.lat_rad = np.radians(lat_deg)
        self.lon_rad_initial = np.radians(lon_deg)
        self.alt_km = alt_km
        self.min_elevation_deg = min_elevation_deg
