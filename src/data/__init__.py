

# Apis
try:
    from .europe_copernicus import OrthoCropEuropeCopernicusApi
except ImportError:
    print('Warning: Could not import OrthoCropEuropeCopernicusApi')

try:
    from .denmark import OrthoCropDenmarkApi
except ImportError:
    print('Warning: Could not import OrthoCropDenmarkApi')

try:
    from .netherlands import OrthoCropNetherlandsApi
except ImportError:
    print('Warning: Could not import OrthoCropNetherlandsApi')

try:
    from .france import OrthoCropFranceApi
except ImportError:
    print('Warning: Could not import OrthoCropFranceApi')

try:
    from .spain import OrthoCropSpainApi
except ImportError:
    print('Warning: Could not import OrthoCropSpainApi')

try:
    from .sentinel import OrthoCropSentinelApi
except:
    print('IMpossbiler to import Sentinlel API')

# Dataset
from .dataset_cropper import OrthoCropDataset

__all__ = [
    "OrthoCropEuropeCopernicusApi", "OrthoCropDenmarkApi", "OrthoCropNetherlandsApi", "OrthoCropFranceApi", "OrthoCropSpainApi",
    "OrthoCropDataset"]