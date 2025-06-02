from ..utils_api import OrthoCropWebApi

class OrthoCropNetherlandsApi(OrthoCropWebApi):

    def __init__(self, **kwarg):

        # Crs Options : ['EPSG:28992', 'EPSG:25832', 'EPSG:4258', 'EPSG:4326', 'EPSG:25831', 'EPSG:3857']
        # Layers : ['Actueel_ortho25IR', '2025_orthoHRIR', '2024_ortho25IR', '2023_ortho25IR', '2022_ortho25IR', '2020_ortho25IR', '2019_ortho25IR', '2018_ortho25IR', '2017_ortho25IR', '2016_ortho25IR']

        layer = kwarg.pop('layer', '2024_orthoHR') # 'Actueel_ortho25IR'
        epsg_query = kwarg.pop('epsg_query', 'EPSG:28992')
        img_format = kwarg.pop('img_format', 'image/png')

        super().__init__(
            wms_url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0",
            wms_version = '1.3.0',
            epsg_query=epsg_query,
            layer = layer,
            img_format = img_format,
            **kwarg,
        )
