"""Stratigraphy Column class"""
import default_config
import moonpies as mp

class StratColumn:
    """
    Class representation of a stratigraphy column.
    """
    def __init__(self, name, lat, lon, age, cfg=None):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.age = age
        self.cfg = cfg if cfg is not None else default_config.Cfg()
        self.time = mp.get_time_array(self.cfg)

        # Init strat columns
        self.ejecta = self._get_ejecta_col()
        self.ice_sources = self._init_ice_sources()

        self.ice = self._get_ice_col()


    def __repr__(self):
        return f'StratColumn({self.name!r},{self.lat}, {self.lon}, {self.age}'


    def __str__(self):
        return self._get_summary()

    def _get_summary(self):
        """Return summary of ice and ejecta columns in self."""
        summary = f'StratColumn {self.name} '
        summary += f'({self.lat:.2f}N, {self.lon:.2f}E) '
        summary += f'{self.age/1e9:.2f} Ga'
        summary += f'Total ice in column: {np.mean(self.ice):.3f} m\n'
        for source, ice_col in self.ice_sources.items():
            summary += f'\t{source}: {np.mean(ice_col):.3f} m\n'
        summary += f'Total ejecta in column: {np.mean(self.ejecta):.3f}'
        return summary


    def _init_ice_sources(self):
        """Store all ice source arrays (m vs. time) in self.ice_sources."""
        self.ice_sources['solar wind'] = mp.solar_wind_ice(self.time, self.cfg)
        self.ice_sources['volcanic'] = mp.volcanic_ice(self.time, self.cfg)
        for r in self.cfg.impact_regimes:
            key = f'impact {r}'
            self.ice_sources[key] = mp.impact_ice(self.time, self.cfg, r)


    def _get_ejecta_col(self):
        """Return ejecta thicknesses at each self.time."""
        return mp.get_ejecta_time(self.time, self.cfg)


    def _get_ice_col(self):
        """Return MoonPIES generated ice_col given ejecta and ice_sources."""
        return mp.run(self.time, self.ejecta, self.ice_sources, self.cfg)


    def _get_strat_col(self):
        """Return stratigraphy column of merged ice and ejecta cols."""
        return mp.make_strat_col(self.time, self.ice, self.ejecta, self.ejecta_sources, self.cfg)
    
    def save():
        pass

    def plot():
        pass