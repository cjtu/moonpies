"""
Thermal equillibrium model to compute ballistic sedimentation melting fraction.

:Authors: K. M. Luchsinger, C. J. Tai Udovicic
"""
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path
import numpy as np
import pandas as pd
from moonpies import config

CFG = config.Cfg()
G_MOON = CFG.grav_moon  # [m/s^2]
R_MOON = CFG.rad_moon  # [m]


def make_melt_frac_dfs(mrs, ej_temps, nseed=10, parallel=True, savepath=""):
    """
    Return mean, stdev DataFrame of melt fractions at each mr and ej_temp.

    Runs nseed times for each mr, ejecta_t combination, randomizing
    initial positions of ejecta pixels. If savepath is given, save as csvs.

    Parameters
    ----------
    mrs (array): Array of mixing ratios [target]
    ej_temps (array): Array of ejecta temperatures
    nseed (int): Number of seeds to run for each mr, ejecta_t combination
    parallel (bool): Whether to run in parallel
    savepath (str): Path to save csv files
    """
    ncpus = cpu_count() if parallel else 1
    seeds = np.arange(nseed)
    melt_frac = get_melt_frac_arr(mrs, ej_temps, seeds, ncpus)
    mf_mean = np.mean(melt_frac, axis=2)
    mf_std = np.std(melt_frac, axis=2)
    df_mean = pd.DataFrame(mf_mean, index=mrs, columns=ej_temps)
    df_std = pd.DataFrame(mf_std, index=mrs, columns=ej_temps)
    if savepath:
        fmean = Path(savepath) / "ballistic_sed_frac_melted_mean.csv"
        fstd = Path(savepath) / "ballistic_sed_frac_melted_std.csv"
        df_mean.to_csv(fmean, float_format="%g")
        df_std.to_csv(fstd, float_format="%g")
        print(f"Wrote to {fmean} and {fstd}")
    return df_mean, df_std


def get_melt_frac_arr(mrs, ejecta_temps, seeds, ncpus=1):
    """
    Return melt fraction array for each mr and ejecta_temp.
    """
    # Sequential triple loop or parallelize on mr
    if not ncpus or ncpus <= 1:
        mf = np.zeros((len(mrs), len(ejecta_temps), len(seeds)))
        for i, mr in enumerate(mrs):
            for j, ejecta_t in enumerate(ejecta_temps):
                for k, seed in enumerate(seeds):
                    mf[i, j, k] = melt_frac_teq_1d(mr, ejecta_t, rng=seed)
    else:
        with Pool(processes=ncpus) as pool:
            args = [(mr, ejecta_temps, seeds) for mr in mrs]
            mf = np.stack(pool.starmap(_parallel_melt_frac, args))
    return mf


def _parallel_melt_frac(mr, ejecta_temps, seeds):
    """Return melt_frac_teq_1d(mr, ej_t, rng=seed) with fixed mr."""
    mf = np.zeros((len(ejecta_temps), len(seeds)))
    for j, ejecta_t in enumerate(ejecta_temps):
        for k, seed in enumerate(seeds):
            mf[j, k] = melt_frac_teq_1d(mr, ejecta_t, rng=seed)
    return mf


def melt_frac_teq_1d(
    mr,
    ejecta_t0,
    target_t0=45,
    melt_t=110,
    n=1000,
    eq_thresh=1,
    max_iter=10**5,
    conv_warning=True,
    rng=None,
):
    """
    Return fraction heated above melt_t from 1D thermal conduction model.

    Model initialized as array of length n with target_t0 at all indices. A
    fraction of random ejecta elements (1-mr) are assigned ejecta_t0. Each
    element is assumed to be 100 microns wide and time step is 1 msec.
    Complete melting is assumed to occur at melt_t with no heat loss.

    The model is stepped through time until one of these conditions is met:
    - Thermal equilibrium: All elements are within eq_thresh of their neighbors
    - All melted: All elements are above melt_t
    - Melting finished: All elements are below melt_t
    - Timeout: The model reaches max_iter without triggering any of the above

    Parameters
    ----------
    ejecta_t0 (num): Initial temperature of the ejecta (pre-impact) [K]
    target_t0 (num): Initial temperature of the target material [K]
    mr (num): Mixing ratio of target material : ejecta material
    melt_t (num): Temperature at which elements are considered melted [K]
    n (num): Length of 1D model [100 micron pixels]
    eq_thresh (num): Threshold temperature for thermal equilibrium [K]
    max_iter (num): Max number of iterations to run (100000 approx. 15 sec)
    conv_warning (bool): Whether to warn if the model does not converge
    rng (int or np.RandomState): Random number generator or seed

    Returns
    -------
    melt_fraction (float): Fraction of target material heated above melt_t
    """
    # Conversion from mixing ratio to ejecta fraction is 1/(1+mr)
    ej_n = int(n * (1 / (1 + mr)))  # Ejecta pixels is ejecta frac * n
    if ej_n == n:  # All pixels ejecta so don't need to run model
        return float(ejecta_t0 >= melt_t)
    ej_inds = rand_positions(ej_n, n, rng=rng)  # Get random ejecta positions

    # Initialize model temperature array with ejecta pixels inserted randomly
    temps = np.ones(n) * target_t0  # Init to target temp
    temps[ej_inds] = ejecta_t0  # Insert ejecta pixels
    melted = np.zeros(n, dtype=bool)  # Init to all not melted
    melted[ej_inds] = True  # Ejecta excluded (set melted but ignore later)
    i = 0
    while (
        i < max_iter
        and not melted.all()  # No convergence after max_iter
        and not (temps < melt_t).all()  # All melted
        and not (  # No more melting (all below melt_t)
            np.abs(np.diff(temps)) < eq_thresh
        ).all()  # Equilibrium
    ):
        # Step through model
        temps = heat1d_finite_diff(temps)

        # Track melted pixels each iteration (once melted, pixel stays melted)
        melted[temps >= melt_t] = True
        i += 1
    if conv_warning and i == max_iter:
        warnings.warn(
            f"Timeout: melt_frac_teq_1d(mr={mr:.3g},ejecta_t0={ejecta_t0},"
            + f"target_t0={target_t0}) quit after {max_iter} iterations"
            + f" (model_time={max_iter*1e-3:.4g} s)."
        )
    total_melted = np.sum(melted) - ej_n  # Count melted, exclude ejecta
    return total_melted / (n - ej_n)  # Frac of target melted


def heat1d_finite_diff(T, density=1800, dt=0.001, dx=1e-5):
    """
    Return updated T from finite difference 1D heat flow conduction model.

    See Eq. 2, Onorato et al. (1978).

    Parameters
    ----------
    T (num or array): temperature [K]
    density (num): density, [kg/m^3]
    k (num): conductivity [W/m/K]
    Cp (num): heat capacity [J/kg/K]
    T_liq_c (num): liquidus of clasts [K]
    T_sol_c (num): solidus of clasts [K]
    dt (num): time step size [s], must match dx
    dx (num): spatial step size [m], must match dt

    Returns
    -------
    alpha (num): scaling factor [unitless]
    """
    K = thermal_diffusivity(thermal_conductivity(T), sphc(T), density)
    c = (K * dt) / dx**2  # coefficient array for finite difference

    # Boundary conditions (assume closed system, link ends)
    t0 = c[0] * T[-1] + (1 - 2 * c[0]) * T[0] + c[0] * T[1]
    tn = c[-1] * T[-2] + (1 - 2 * c[-1]) * T[-1] + c[-1] * T[0]

    # Compute T at next timestep for main array
    T[1:-1] = c[1:-1] * T[:-2] + (1 - 2 * c[1:-1]) * T[1:-1] + c[1:-1] * T[2:]

    # Set boundary conditions
    T[0] = t0
    T[-1] = tn
    return T


def thermal_diffusivity(conductivity, cp, density=1800):
    """Return thermal diffusivity."""
    return conductivity / (density * cp)


def thermal_conductivity(T, Kc=3.4e-3, chi=2.7):
    """Return thermal conductivity of regolith (eqn A4 Hayne 2017)."""
    return Kc * (1 + chi * (T / 350) ** 3)


def sphc(T, c0=-3.6125, c1=2.7431, c2=2.3616e-3, c3=-1.234e-5, c4=8.9093e-9):
    """Return specific heat capacity [J/kg/K] of regolith."""
    return c0 + c1 * T + c2 * T**2 + c3 * T**3 + c4 * T**4


def rand_positions(ej_n, n=100, rng=None):
    """
    Return ej_n random initial positions in target of length n.

    Parameters
    ----------
    ej_n (int): Number of ejecta pixels to assign
    n (int): Length of 1D target [pixels]
    rng (int or np.RandomState): Random number generator or seed

    Returns
    -------
    inds (array): Indices of initial positions of ejecta
    """
    rng = _rng(rng)
    inds = rng.choice(np.arange(n), size=ej_n, replace=False)
    return inds


def _rng(rng):
    """Shorthand wrapper for np.random.default_rng."""
    return np.random.default_rng(rng)


if __name__ == "__main__":
    # Sample target fraction range [0, 1), convert to mixing ratio
    # Maximum mixing ratio on Moon ~32 (using Petro and Pieters, 2006)
    # The following results in mrs of [0, 39]
    FRACS = np.linspace(0, 1, 40, endpoint=False)  # Mixing frac (target/total)
    MRS = FRACS / (1 - FRACS)  # Mixing ratio (target:ejecta)
    TEMPS = np.linspace(110, 510, 81)  # Ejecta temperatures [K]
    NSEED = 20  # random samples for each combination

    # Generate melt_frac csvs
    make_melt_frac_dfs(MRS, TEMPS, NSEED, savepath=CFG.data_path)
