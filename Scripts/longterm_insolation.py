"""
Long-term 65N integrated insolation (Huybers-style ISI) from the Laskar et al. (2004)
orbital solution, for forcing emulator rollouts beyond the observational record.

Also 65N June-solstice insolation (the Laskar counterpart of the Berger 1990 'insol_ber90' record), which
run_config uses as the 'insol_la04' insolation record.

Usage (from a notebook):
    import longterm_insolation as li
    years, long_R = li.compute_long_insolation(start_kaBP=1462, end_kyr=10000)
    Q = li.solstice_insolation(np.arange(-1461, 10001))   # kyr from J2000, negative = past

or from the command line, to compute once and cache to Data/:
    python longterm_insolation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from climlab.solar.insolation import daily_insolation

PROJECT = Path(__file__).resolve().parents[1]
LA2004_CSV = PROJECT / 'Data' / 'Observations' / 'laskar2004' / 'INSOL.LA2004.BTL.csv'


def load_la2004(path=LA2004_CSV):
    """
    Laskar et al. (2004) orbital solution: eccentricity, obliquity (rad), and
    longitude of perihelion (rad, unwrapped/continuous) vs. time from J2000
    (kyr). Covers t in [-51000, +21000], i.e. plenty of past AND future -
    unlike climlab's bundled OrbitalTable (Berger & Loutre 1991), which only
    goes back to -5000 kyr and has NO future (kyear > 0) values at all.
    """
    return pd.read_csv(path, comment='#', header=None, names=['t', 'ecc', 'obliq', 'varpi'])


def solstice_insolation(t_kyr, lat=65, la2004=None):
    """
    Daily-mean insolation [W/m2] at latitude `lat` on the June solstice (solar longitude 90 deg), at times
    t_kyr [kyr from J2000; negative = past, positive = future] from the Laskar et al. (2004) solution.
    Laskar counterpart of the Berger (1990) Q65N_solstice_raw record (r = 0.98 with it over 0-1461 ka).
    Vectorized: one climlab call for all times.
    """
    if la2004 is None:
        la2004 = load_la2004()
    t_kyr = np.asarray(t_kyr, dtype=float)
    if t_kyr.min() < la2004['t'].min() or t_kyr.max() > la2004['t'].max():
        raise ValueError(f"times must lie within the Laskar 2004 solution ({la2004['t'].min():.0f} to "
                         f"{la2004['t'].max():.0f} kyr)")

    # interpolated in radians (varpi is continuous in the source file), then converted to degrees and wrapped
    # to [0, 360), as in compute_long_insolation
    orb = {
        'ecc': np.interp(t_kyr, la2004['t'], la2004['ecc']),
        'obliquity': np.degrees(np.interp(t_kyr, la2004['t'], la2004['obliq'])),
        'long_peri': np.degrees(np.interp(t_kyr, la2004['t'], la2004['varpi'])) % 360,
    }
    # day_type=2: `day` is the solar longitude [deg], so 90 is the June solstice exactly in every year
    return np.asarray(daily_insolation(lat=lat, day=90, orb=orb, day_type=2), dtype=float).reshape(t_kyr.shape)


def compute_long_insolation(start_kaBP=1462, end_kyr=10000, lat=65, tau=0, la2004=None):
    """
    Integrated insolation (GJ/m2) on a 1-kyr grid from -start_kaBP to +end_kyr
    (negative = past, positive = future, matching the Laskar time axis).

    start_kaBP - oldest age needed (the old notebook used len(data) + 1)
    tau        - daily-insolation threshold in W/m2 (Huybers 2006). NOTE: tau=0 sums
                 every day, i.e. annual insolation, which has no precession signal.
                 It must match the threshold of the training forcing (ISI_thresh0Wm2).

    returns (years, long_R)
    """
    if la2004 is None:
        la2004 = load_la2004()

    years = np.arange(-int(start_kaBP), int(end_kyr) + 1, dtype=float)

    # Interpolate orbital elements onto the target years. obliq/varpi are interpolated in radians
    # (varpi is left unwrapped in the source file) and only converted to degrees + wrapped to [0, 360)
    # afterward, to avoid interpolation artifacts at the 0/360 boundary.
    ecc_interp = np.interp(years, la2004['t'], la2004['ecc'])
    obliquity_interp = np.degrees(np.interp(years, la2004['t'], la2004['obliq']))
    long_peri_interp = np.degrees(np.interp(years, la2004['t'], la2004['varpi'])) % 360

    days_of_year = np.arange(1, 366)

    long_R = np.zeros_like(years)
    for i in tqdm(range(len(years)), desc="Computing ISI"):
        orb_i = {
            'ecc': ecc_interp[i],
            'obliquity': obliquity_interp[i],
            'long_peri': long_peri_interp[i],
        }
        W = np.asarray(daily_insolation(lat=lat, day=days_of_year, orb=orb_i))
        J = np.sum(W[W >= tau] * 86400)  # joules/m2, summed over qualifying days
        long_R[i] = J / 1e9  # GJ/m2, matching Huybers' reported units

    return years, long_R


if __name__ == "__main__":
    years, long_R = compute_long_insolation()
    out = PROJECT / 'Data' / 'Observations' / 'laskar2004' / 'ISI_65N_tau0_LA2004.csv'
    pd.DataFrame({'t_kyr_from_J2000': years, 'ISI[GJ/m2]': long_R}).to_csv(out, index=False)
    print("saved", out)
