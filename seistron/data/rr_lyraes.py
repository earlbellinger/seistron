"""module for rr lyrae data parsing"""

import pandas as pd
from .data import StellarData
from seistron.utils.labels import PERIOD_LABEL, PERIOD_SHORT_LABEL, AMPLITUDE_LABEL, AMPLITUDE_SHORT_LABEL

def load_rrlyrae() -> StellarData:
    # load the rr lyrae data as a pandas dataframe
    df = pd.DataFrame()  # replace with actual data loading code
    metadata = {
        "period": {"label": PERIOD_LABEL, "short_label": PERIOD_SHORT_LABEL},
        "amplitude": {"label": AMPLITUDE_LABEL, "short_label": AMPLITUDE_SHORT_LABEL},
        # add additional columns as needed
    }
    return StellarData(df, metadata)
