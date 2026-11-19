"""
65N insolation from Laskar et al. (2004) orbital elements (INSOL.LA2004.BTL).

Assumed columns: t (kyr from J2000, negative = past), ecc, obliq (rad),
varpi (rad, longitude of perihelion from the moving equinox, heliocentric,
unwrapped). Check the convention on the t = 0 row before trusting output:
    ecc ~ 0.0167, obliq ~ 0.4091 rad, varpi mod 2*pi ~ 1.796 rad (102.9 deg)
If varpi mod 2*pi is instead ~ 4.94 rad (282.9 deg), the file already uses
the geocentric convention: set GEOCENTRIC = True below.

Only numpy/pandas are used; all formulas are the standard daily-mean
insolation expressions (e.g. Berger 1978).
"""
import numpy as np
import pandas as pd

S0 = 1365.0        # W/m^2, only sets the absolute level (irrelevant after standardising)
YEAR = 365.2422    # days
D_VE = 80.0        # vernal equinox fixed at March 21 (day of year, Jan 1 = day 1)
GEOCENTRIC = False


def _lam_perihelion(varpi):
    """Geocentric solar longitude at perihelion (rad)."""
    return varpi if GEOCENTRIC else varpi + np.pi


def daily_insolation(lat_deg, lam, e, eps, varpi):
    """Daily-mean TOA insolation (W/m^2) at true solar longitude lam (rad, 0 = vernal equinox)."""
    phi = np.deg2rad(lat_deg)
    delta = np.arcsin(np.sin(eps) * np.sin(lam))                      # solar declination
    a_over_r = (1.0 + e * np.cos(lam - _lam_perihelion(varpi))) / (1.0 - e**2)
    x = -np.tan(phi) * np.tan(delta)
    h0 = np.arccos(np.clip(x, -1.0, 1.0))                              # clip handles polar day/night
    return S0 / np.pi * a_over_r**2 * (h0 * np.sin(phi) * np.sin(delta)
                                       + np.cos(phi) * np.cos(delta) * np.sin(h0))


def lambda_of_day(day, e, varpi, n_iter=12):
    """True solar longitude (rad) on calendar day `day`, vernal equinox fixed at D_VE."""
    lam_p = _lam_perihelion(varpi)
    nu_ve = -lam_p                                                     # true anomaly at the equinox
    E_ve = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu_ve / 2), np.sqrt(1 + e) * np.cos(nu_ve / 2))
    M = (E_ve - e * np.sin(E_ve)) + 2 * np.pi * (day - D_VE) / YEAR   # mean anomaly on `day`
    E = M.copy() if isinstance(M, np.ndarray) else np.array(M, dtype=float)
    for _ in range(n_iter):                                            # Kepler: E - e sin E = M
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    return nu + lam_p


def solstice_insolation(lat_deg, e, eps, varpi):
    """Daily-mean insolation at the NH summer solstice (lam = 90 deg)."""
    return daily_insolation(lat_deg, np.pi / 2, e, eps, varpi)


def monthly_mean_insolation(lat_deg, e, eps, varpi, first_day=182, last_day=212):
    """Mean insolation over calendar days first_day..last_day (default: July, 365-day year)."""
    days = np.arange(first_day, last_day + 1, dtype=float)[:, None] + 0.5   # noon of each day
    lam = lambda_of_day(days, e[None, :], varpi[None, :])
    return daily_insolation(lat_deg, lam, e[None, :], eps[None, :], varpi[None, :]).mean(axis=0)


if __name__ == "__main__":
    # Adjust the path/delimiter to however the file loads for you.
    orb = pd.read_csv("INSOL.LA2004.BTL.csv")
    t, e, eps, varpi = orb.iloc[:, :4].to_numpy(dtype=float).T

    print("t=0 row:", orb[t == 0].to_string(index=False))
    print("varpi(t=0) mod 2pi [deg]:", np.rad2deg(np.mod(varpi[t == 0], 2 * np.pi)))

    keep = (t >= -1000) & (t <= 0)          # last 1 Myr (spin-up from 1 Ma, as SM91 did)
    age = -t[keep]                          # kyr BP
    out = pd.DataFrame({
        "age_kyr": age,
        "Q65N_july": monthly_mean_insolation(65, e[keep], eps[keep], varpi[keep]),
        "Q65N_solstice": solstice_insolation(65, e[keep], eps[keep], varpi[keep]),
    }).sort_values("age_kyr", ascending=False)   # past -> present
    out.to_csv("insolation_65N_la2004.csv", index=False)
    print(out.head())
    print(out.tail())