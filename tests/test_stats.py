"""Stats helpers."""
import numpy as np
import pandas as pd

from analysis.stats import holm


def test_holm_matches_hand_computation():
    p = pd.Series([0.01, 0.04, 0.03, 0.20], index=list("abcd"))
    # sorted 0.01, 0.03, 0.04, 0.20 -> 0.04, 0.09, 0.09 (monotone), 0.20
    np.testing.assert_allclose(holm(p).loc[list("acbd")].to_numpy(), [0.04, 0.09, 0.09, 0.20])
    q = holm(pd.Series([0.5, np.nan, 0.01]))
    assert np.isnan(q.iloc[1]) and q.iloc[2] == 0.02 and q.iloc[0] == 0.5
