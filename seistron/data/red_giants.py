"""module for red giant data parsing"""

import pandas as pd
from .data import StellarData
from seistron.utils.labels import L_LABEL, L_SHORT_LABEL, TEFF_LABEL, TEFF_SHORT_LABEL, FEH_LABEL, FEH_SHORT_LABEL

def load_yu2018() -> StellarData:
    # load the ji yu 2018 red giants data as a pandas dataframe
    df = pd.DataFrame()  # replace with actual data loading code
    metadata = {
        "L": {"label": L_LABEL, "short_label": L_SHORT_LABEL},
        "Teff": {"label": TEFF_LABEL, "short_label": TEFF_SHORT_LABEL},
        "[Fe/H]": {"label": FEH_LABEL, "short_label": FEH_SHORT_LABEL},
        # add additional columns as needed
    }
    return StellarData(df, metadata)
