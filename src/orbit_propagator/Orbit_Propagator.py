class OrbitPropagator:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self.satellite = self._load_satellite()
