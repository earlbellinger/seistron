"""module for stellar cluster data parsing"""

import pandas as pd
from .data import StellarData
from seistron.utils.labels import AGE_LABEL, AGE_SHORT_LABEL, DISTANCE_LABEL, DISTANCE_SHORT_LABEL

def load_clusters() -> StellarData:
    # load the stellar cluster data as a pandas dataframe
    df = pd.DataFrame()  # replace with actual data loading code
    metadata = {
        "age": {"label": AGE_LABEL, "short_label": AGE_SHORT_LABEL},
        "distance": {"label": DISTANCE_LABEL, "short_label": DISTANCE_SHORT_LABEL},
        # add additional columns as needed
    }
    return StellarData(df, metadata)