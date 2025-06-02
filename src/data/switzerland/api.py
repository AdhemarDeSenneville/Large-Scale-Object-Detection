from ..utils_api import OrthoCropWebApi, OrthoPyramidWebApi

class OrthoCropSwitzerlandApi(OrthoCropWebApi):

    def __init__(self, **kwarg):

        # ['EPSG:32631', 'EPSG:4230', 'CRS:84', 'EPSG:25831', 'EPSG:3857', 'EPSG:32630', 'EPSG:4083', 'EPSG:25829', 'EPSG:23029', 'EPSG:25830', 'EPSG:4082', 'EPSG:32629', 'EPSG:4081', 'EPSG:23030', 'EPSG:23031', 'EPSG:25828', 'EPSG:4258', 'EPSG:4326', 'EPSG:32628', 'EPSG:4080']
        # Layers : ['OI.MosaicElement', 'OI.OrthoimageCoverage', 'fondo']

        raise NotImplementedError
        layer = kwarg.pop('layer', 'OI.MosaicElement')
        epsg_query = kwarg.pop('epsg_query', 'EPSG:25829') # Change epsg
        img_format = kwarg.pop('img_format', 'image/png')

        super().__init__(
            wms_url = "http://www.ign.es/wms-inspire/pnoa-ma",
            wms_version = '1.3.0',
            epsg_query=epsg_query,
            layer = layer,
            img_format = img_format,
            **kwarg,
        )




class OrthoPyramidSwitzerlandApi(OrthoPyramidWebApi):

    def __init__(self, **kwarg):

        # ['EPSG:32631', 'EPSG:4230', 'CRS:84', 'EPSG:25831', 'EPSG:3857', 'EPSG:32630', 'EPSG:4083', 'EPSG:25829', 'EPSG:23029', 'EPSG:25830', 'EPSG:4082', 'EPSG:32629', 'EPSG:4081', 'EPSG:23030', 'EPSG:23031', 'EPSG:25828', 'EPSG:4258', 'EPSG:4326', 'EPSG:32628', 'EPSG:4080']
        # Layers : ['OI.MosaicElement', 'OI.OrthoimageCoverage', 'fondo']

        raise NotImplementedError
