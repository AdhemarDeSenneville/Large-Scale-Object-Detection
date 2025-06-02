from ..utils_api import OrthoCropWebApi

class OrthoCropFranceApi(OrthoCropWebApi):

    def __init__(self, **kwarg):

        # 
        # 
        layer = kwarg.pop('layer', 'ORTHOIMAGERY.ORTHOPHOTOS.BDORTHO')
        epsg_query = kwarg.pop('epsg_query', 'EPSG:2154')
        img_format = kwarg.pop('img_format', 'image/png')

        super().__init__(
            wms_url = "https://data.geopf.fr/wms-r",
            wms_version = '1.3.0',
            epsg_query=epsg_query,
            layer = layer,
            img_format = img_format,
            **kwarg,
        )


        