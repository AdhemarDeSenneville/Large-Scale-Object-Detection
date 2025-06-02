from ..utils_api import OrthoCropWebApi

# Implement the Pyramid API (256 256 crops)

class OrthoCropEuropeCopernicusApi(OrthoCropWebApi):

    def __init__(self, **kwarg):

        # Crs Options : ['EPSG:4326', 'EPSG:3035', 'CRS:84']
        # Layers : ['VHR_2021_LAEA']


        layer = kwarg.pop('layer', 'VHR_2021_LAEA')
        epsg_query = kwarg.pop('epsg_query', 'EPSG:3035')
        img_format = kwarg.pop('img_format', 'image/geotiff')


        super().__init__(
            wms_url = "https://copernicus.discomap.eea.europa.eu/arcgis/services/GioLand/VHR_2021_LAEA/ImageServer/WMSServer",
            wms_version = '1.3.0',
            epsg_query=epsg_query,
            layer = layer,
            img_format = img_format,
            **kwarg,
        )


