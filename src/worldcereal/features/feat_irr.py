# Some additional dedicated features specifically for
# irrigation detection
import math

import hdmedians as hd
import numba
import numexpr as ne
import numpy as np
from satio.features import Features
from satio.timeseries import Timeseries
from scipy.signal import argrelextrema, resample




###################################################################
# specific vegetation indices
###################################################################

def calc_gvmi(B08, B12):
    return ((B08+0.1)-(B12+0.02))/((B08+0.1)+((B12+0.02)))


def calc_ndwi_veg(B08, B12):
    return (B08-B12)/(B08+B12)


def calc_str_1(B11):
    B11 = B11 * 10000
    B11 = B11.astype(np.int16)
    str_1 = ((1-(B11/65500))**2)/(2*(B11/65500))
    return str_1.astype(np.float32)


###################################################################
# additional features based on time series...
###################################################################

def find_peaks(x):
    x = np.where(x < 0, 0, x)
    local_max = argrelextrema(x, np.greater)
    local_max = np.array(local_max[0])
    peaks = len(local_max)
    return (peaks)


def find_maxdate(x, composite_freq):
    max_ = np.nanmax(x)
    if np.isnan(max_):
        return 0
    else:
        idx = np.argwhere(x == max_).flatten()[0]
        duration = idx * composite_freq
        return duration


def cumtsteps(ts, n_steps=6):
    # Calculate cumulative timeseries
    ts_cumsum = np.cumsum(ts, axis=0)

    return resample(ts_cumsum, n_steps, axis=0)


def cumfeat(ts, composite_freq=10):

    # Calculate cumulative timeseries
    ts_cumsum = np.cumsum(ts, axis=0)

    # Find amount of local maxima in cumulated timeseries
    n_peaks = np.apply_along_axis(find_peaks, 0, ts_cumsum)

    # Calculate maximum and minimum of cumulative timeseries
    max_cum = np.nanmax(ts_cumsum, axis=0)
    min_cum = np.nanmin(ts_cumsum, axis=0)

    # Calculate the duration till the maximum of cumulative timeseries
    duration = np.apply_along_axis(find_maxdate, 0, ts_cumsum,
                                   composite_freq)

    # Calculate the slope between the start and maximum
    # cumulative timeseries
    slope = np.arctan(np.divide(max_cum, duration,
                                out=np.zeros_like(max_cum),
                                where=duration != 0)) * 180 / np.pi

    # Merge all the 2D features in a 3D numpy array
    features = np.array([max_cum, min_cum, n_peaks, duration, slope])

    return features


def sum_div(x, div=1):
    return np.sum(x, axis=0, keepdims=True) / div


###################################################################
# RELATED TO SMAD FEATURE
###################################################################

@numba.njit
def mean_numba(a):
    """
    Replacement of np.mean() function that does run with Numba
    :Parameters:
     - `a` (np.array) - list of values (2d numpy array)
    :Return:
     - np.array(mean): the mean of axis=1 (np.mean(a,axis=1))
       of a specific 2d-array;
    """
    res = []
    for i in numba.prange(a.shape[0]):  # pylint: disable=not-an-iterable
        res.append(a[i, :].mean())
    return np.array(res)


@numba.njit
def isin_numba(a, b):
    """
    Replacement of np.isnan() function that does run with Numba
    :Parameters:
     - `a` (list|np.array) - list of values
     - `b` (float) - value to check if exists in a
    :Return:
     - bool: True or False;
    """
    sim = False
    for i in numba.prange(len(a)):  # pylint: disable=not-an-iterable
        if a[i] == b:
            sim = True
    return sim


@numba.njit
def geometric_median(X):
    """
    Compute the geometric median of a point sample.
    The geometric median coordinates will be expressed
    in the Spatial Image reference system (not in real world metrics).
    We use the Weiszfeld's algorithm
    (http://en.wikipedia.org/wiki/Geometric_median)
    :Parameters:
     - `X` (list|np.array) - voxels coordinate (6xN matrix)
     - `numIter` (int) - limit the length of the search for global optimum
    :Return:
     - np.array(gm_B02,gm_B03,gm_B04,gm_B08,gm_B11,gm_B12):
        geometric median per band;
    """
    numIter = 200

    X_ = X.T[np.sum(X.T, axis=1) > 0].T  # remove nodata values
    y = np.full(6, np.nan, dtype=np.float32)  # prepare output dataset
    if len(X_[0]) > 0:

        # -- Initialising 'median' to the centroid
        y = mean_numba(X_)

        # -- Check if init point in set of points
        t0 = isin_numba(X_[0], y[0])
        t1 = isin_numba(X_[1], y[1])
        t2 = isin_numba(X_[2], y[2])
        t3 = isin_numba(X_[3], y[3])
        t4 = isin_numba(X_[4], y[4])
        t5 = isin_numba(X_[5], y[5])

        # -- If the init point is in the set of points, we shift it:
        while t0 and t1 and t2 and t3 and t4 and t5:
            y += 0.01

        # boolean testing the convergence toward a global optimum
        convergence = False

        # list recording the distance evolution
        dist = [0.0]

        # -- Minimizing the sum of the squares of the distances between
        # each points in 'X_' and the median.
        i = 0
        while ((not convergence) and (i < numIter)):
            num_02, num_03, num_04, num_08, num_11, num_12 = (0.0, 0.0, 0.0,
                                                              0.0, 0.0, 0.0)
            denum = 0.0
            m = X_.shape[1]
            d = 0
            for j in numba.prange(m):  # pylint: disable=not-an-iterable
                div = math.sqrt((X_[0, j]-y[0])**2 +
                                (X_[1, j]-y[1])**2 + (X_[2, j]-y[2])**2 +
                                (X_[3, j]-y[3])**2 +
                                (X_[4, j]-y[4])**2 + (X_[5, j]-y[5])**2)
                num_02 += X_[0, j] / div
                num_03 += X_[1, j] / div
                num_04 += X_[2, j] / div
                num_08 += X_[3, j] / div
                num_11 += X_[4, j] / div
                num_12 += X_[5, j] / div
                denum += 1./div
                d += div**2  # distance (to the median) to miminize

            dist.append(d)  # update of the distance evolution
            dist = [d for d in dist if d > 0]  # remove initial value

            if denum == 0.:
                return np.full(6, 0., dtype=np.float32)

            y[0] = num_02/denum
            y[1] = num_03/denum
            y[2] = num_04/denum
            y[3] = num_08/denum
            y[4] = num_11/denum
            y[5] = num_12/denum

            if i > 3:
                # we test the convergence over three steps for stability
                convergence = (abs(dist[i]-dist[i-2]) < 0.001)

            i += 1

        return y

    else:
        return


def to_stack(x):
    """
    Transpose a band timeseries (t,x,y) to a 2d-array (i,t)
    with i representing
    a coordinate (x,y) and t representing the timestep
    :Parameters:
     - `x` (np.array) - image timeseries expressed as a 3d numpy array
    :Return:
     - np.array(((i_0,t_0),(i_1,t_0),...,(i_n,t_0))((i_0,t_1),(i_1,t_1),...,
     (i_n,t_1)_))...((i_0,t_n),(i_1,t_n),...,(i_n,t_n)_)):
        image timeseries expressed as a 2d-array;
    """
    x = x.transpose(1, 2, 0).reshape(x.shape[1]*x.shape[2], -1)
    return x


def bands2stack(x_02, x_03, x_04, x_08, x_11, x_12):
    """
    Transpose and stack band timeseries to single 3d-array (i,b,t),
        with i representing a coordinate (x,y),
        b representing the band, and t representing the timestep
    :Parameters:
     - `x_02` (np.array) - image timeseries of band 2
     - `x_03` (np.array) - image timeseries of band 3
     - `x_04` (np.array) - image timeseries of band 4
     - `x_08` (np.array) - image timeseries of band 8
     - `x_11` (np.array) - image timeseries of band 11
     - `x_12` (np.array) - image timeseries of band 12
    :Return:
     - np.array(xt_02,xt_03,xt_04,xt_08,xt_11,xt_12):
        stack of transposed band timeseries;
    """
    xt_02 = to_stack(x_02)
    xt_03 = to_stack(x_03)
    xt_04 = to_stack(x_04)
    xt_08 = to_stack(x_08)
    xt_11 = to_stack(x_11)
    xt_12 = to_stack(x_12)
    return np.array([np.array([t2, t3, t4, t8, t11, t12])
                     for t2, t3, t4, t8, t11, t12 in
                     zip(xt_02, xt_03, xt_04, xt_08, xt_11, xt_12)])


def hd_geometric_median(data):

    y = np.mean(data, axis=2)

    data = data.astype('float32')

    for b in range(data.shape[1]):
        band_median = hd.geomedian(data[:, b],
                                   eps=0.001,
                                   maxiters=200)
        y[:, b] = np.array(band_median)

    return y


def calculate_smad(x_02, x_03, x_04, x_08, x_11, x_12):
    """
    Calculate SMAD over 6 Sentinel-2 bands (B02,B03,B04,B08,B11,B12)
    :Parameters:
     - `x_02` (np.array) - image timeseries of band 2
     - `x_03` (np.array) - image timeseries of band 3
     - `x_04` (np.array) - image timeseries of band 4
     - `x_08` (np.array) - image timeseries of band 8
     - `x_11` (np.array) - image timeseries of band 11
     - `x_12` (np.array) - image timeseries of band 12
    :Return:
     - np.array(SMAD,gm_x02,gm_x03,gm_x04,gm_x08,gm_x11,gm_x12):
        calculate SMAD feature and geometric median of every band;
    """
    # stack all the bands to a (i,b,t) format
    stack = bands2stack(x_02, x_03, x_04, x_08, x_11, x_12)

    # stack = stack.astype(np.float32)
    # # vectorize the gemoetric median function
    # vgm = np.vectorize(geometric_median, signature='(i,j)->(a)')
    # # run the vectorized geometric median function for the stacked bands
    # gm_stack = vgm(stack)

    # compute the geometric medians
    gm_stack = hd_geometric_median(stack)

    # get the timesteps and x and y resolution
    ts = x_02.shape[0]
    xs = x_02.shape[1]
    ys = x_02.shape[2]
    # create empty arrays to store the geometric median
    gm_x02 = np.full((xs, ys), np.nan, dtype=np.float32)
    gm_x03 = np.full((xs, ys), np.nan, dtype=np.float32)
    gm_x04 = np.full((xs, ys), np.nan, dtype=np.float32)
    gm_x08 = np.full((xs, ys), np.nan, dtype=np.float32)
    gm_x11 = np.full((xs, ys), np.nan, dtype=np.float32)
    gm_x12 = np.full((xs, ys), np.nan, dtype=np.float32)
    # create an array of coordinates using np.meshgrid
    i_coords, j_coords = np.meshgrid(range(xs), range(ys), indexing='ij')
    coords = np.vstack(np.stack([np.dstack([i, j])[0]
                                 for i, j in zip(i_coords, j_coords)]))

    # store the geometric median per band based on the coordinate position
    for y, c in zip(gm_stack, coords):
        if np.isfinite(y[0]):
            gm_x02[c[0], c[1]] = y[0]
            gm_x03[c[0], c[1]] = y[1]
            gm_x04[c[0], c[1]] = y[2]
            gm_x08[c[0], c[1]] = y[3]
            gm_x11[c[0], c[1]] = y[4]
            gm_x12[c[0], c[1]] = y[5]

    # calculate the SMAD feature
    smad = []
    for i in range(ts):  # calculate the cosdist per timestep
        cosdist = 1 - ((x_02[i] * gm_x02 +
                        x_03[i] * gm_x03 +
                        x_04[i] * gm_x04 +
                        x_08[i] * gm_x08 +
                        x_11[i] * gm_x11 +
                        x_12[i] * gm_x12) /
                       ((np.sqrt((x_02[i]) ** 2 + (x_03[i]) ** 2 +
                                 (x_04[i]) ** 2 + (x_08[i]) ** 2 +
                                 (x_11[i]) ** 2 + (x_12[i]) ** 2)
                         * np.sqrt((gm_x02) ** 2 + (gm_x03) ** 2 +
                                   (gm_x04) ** 2 + (gm_x08) ** 2 +
                                   (gm_x11) ** 2 + (gm_x12) ** 2))))
        smad.append(cosdist)
    smad = np.nanmedian(smad, axis=0)

    features_names = ['smad-med-20m',  'B02-gm-20m', 'B03-gm-20m',
                      'B04-gm-20m', 'B08-gm-20m',
                      'B11-gm-20m', 'B12-gm-20m']
    features = np.stack([smad, gm_x02, gm_x03, gm_x04, gm_x08,
                         gm_x11, gm_x12], axis=0)

    return Features(features, features_names)


###################################################################
# everything related to soil moisture...
# implementation based on NDVI, L8 thermal and agera5 data
###################################################################


CONST = {'zero_celcius': 273.15,  # 0 degrees C in K,

         # reference values
         't_ref': 293.15,  # reference temperature 20 degrees celcius
         'p_ref': 1013.25,  # reference pressure in mbar
         'z_ref': 0,  # sea level m

         'lapse': -0.0065,  # lapse rate K m-1
         'g': 9.807,  # gravity m s-2
         'gc_spec': 287.0,  # gas constant J kg-1 K-1
         'gc_dry': 2.87,  # dry air gas constant mbar K-1 m3 kg-1
         'gc_moist': 4.61,  # moist air gas constant mbar K-1 m3 kg-1
         'r_mw': 0.622,  # ratio water particles/ air particles
         'sh': 1004.0,  # specific heat J kg-1 K-1
         'lh_0': 2501000.0,  # latent heat of evaporation at 0 C [J/kg]
         'lh_rate': -2361,  # rate of latent heat vs temperature [J/kg/C]
         'power': 9.807 / (0.0065 * 287.0),
         'k': 0.41,  # karman constant (-)
         'sol': 1367,  # maximum solar radiation at top of atmosphere W m-2
         'sb': 5.67e-8,  # stefan boltzmann constant
         'day_sec': 86400.0,  # seconds in a day
         'year_sec': 86400.0 * 365,  # seconds in a year

         'absorbed_radiation': 0.48,  # biomass factor
         'conversion': 0.864,  # conv biom. calc. from g s-1 m-2 to kg ha-1 d-1

         'z0_soil': 0.001  # soil roughness m
         }


def monin_obukhov_length(h_flux, ad, u_star, t_air_k):
    sh = CONST['sh']
    k = CONST['k']
    g = CONST['g']
    res = ne.evaluate("(-ad*sh*u_star**3*t_air_k)/(k*g*h_flux)")

    return res


def wet_bulb_temperature_inst(t_air_i, t_dew_i):
    tw = wetbulb_temperature_vec(t_air_i, t_dew_i)

    return tw


@numba.jit(nopython=True)
def dew_point_temperature_inst(vp_i):
    t_dew_i = (237.3 * np.log(vp_i / 6.108)) / (17.27 - np.log(vp_i / 6.108))
    t_dew_i = np.where(np.isfinite(t_dew_i), t_dew_i, np.nan)
    return t_dew_i


def dew_point_temperature_coarse_inst(vp_i):
    t_dew_i = ne.evaluate(("(237.3 * log(vp_i / 6.108)) / "
                           "(17.27 - log(vp_i / 6.108))"))
    t_dew_i = np.where(np.isfinite(t_dew_i), t_dew_i, np.nan)

    return t_dew_i


@numba.jit(nopython=True)
def latent_heat_numba(t):
    lv = 1000 * (2501 - 2.361 * t)
    return lv


@numba.jit(nopython=True)
def psychometric_constant_numba(lv, p=1013.25, cp=1004, rm=0.622):
    psy = (cp * p) / (lv * rm)
    return psy


@numba.jit(nopython=True)
def vapor_pressure_numba(t):
    vp = 6.108 * np.exp((17.27 * t) / (237.3 + t))
    return vp


@numba.vectorize(
    ["float32(float32, float32)", "float64(float64, float64)"],
    nopython=True,
    target="parallel",
)
def wetbulb_temperature_vec(ta, td):
    maxiter = 1000
    tol = 1e-3
    pressure = 1013.25
    lv = latent_heat_numba(ta)
    psy = psychometric_constant_numba(lv, p=pressure)
    tw = td + ((ta - td) / 3)
    ea_ta = vapor_pressure_numba(td)
    n = 0
    prev_dir = 0
    step = (ta - td) / 5.
    while abs(step) > tol:
        ea_tw = vapor_pressure_numba(tw) - psy * (ta - tw)
        direction = (-1) ** ((ea_tw - ea_ta) > 0)
        if prev_dir != direction:
            step *= 0.5
        tw += step * direction
        prev_dir = direction
        n += 1
        if n >= maxiter:
            return np.nan
    return tw


def psi_m(y):
    a = 0.33
    b = 0.41
    pi = np.pi
    x = ne.evaluate("(y/a)**(1./3.)")
    phi_0 = ne.evaluate("(-log(a)+sqrt(3)*b*a**(1./3.)*pi/6.)")
    res = ne.evaluate(
        "(log(a+y)-3*b*y**(1./3.)+(b*a**(1./3.))/2.*log((1+x)**2/"
        "(1-x+x**2))+sqrt(3)*b*a**(1./3.)*arctan((2*x-1)/sqrt(3))"
        "+phi_0)"
    )
    res = np.where(np.isfinite(res), res, np.nan)

    return res


def psi_h(y):
    c = 0.33
    d = 0.057
    n = 0.78
    res = ne.evaluate("((1-d)/n)*log((c+y**n)/c)")
    res = np.where(np.isfinite(res), res, np.nan)

    return res


def initial_friction_velocity_inst(u_b_i, z0m, disp, z_b=100):
    k = CONST['k']
    res = ne.evaluate("(k * u_b_i) / (log((z_b - disp) / z0m))")
    res = np.where(np.isfinite(res), res, np.nan)

    return res


def atmospheric_emissivity_inst(vp_i, t_air_k_i):
    return 1.24 * (vp_i / t_air_k_i) ** (1. / 7.)


def net_radiation_bare(ra_hor_clear_i, emiss_atm_i,
                       t_air_k_i, lst, r0_bare=0.38):
    emiss_bare = 0.95
    sb = CONST['sb']
    rn_bare = ne.evaluate(
        "(1 - r0_bare) * ra_hor_clear_i + emiss_atm_i "
        "* emiss_bare * sb * (t_air_k_i) ** 4 - "
        "emiss_bare * sb * (lst) ** 4")
    return rn_bare


def net_radiation_full(ra_hor_clear_i, emiss_atm_i,
                       t_air_k_i, lst, r0_full=0.18):
    emiss_full = 0.99
    sb = CONST['sb']
    rn_full = ne.evaluate(
        "(1 - r0_full) * ra_hor_clear_i + emiss_atm_i "
        "* emiss_full * sb * (t_air_k_i) ** 4 - emiss_full "
        "* sb * (lst) ** 4")
    return rn_full


def sensible_heat_flux_bare(rn_bare, fraction_h_bare=0.65):
    return rn_bare * fraction_h_bare


def sensible_heat_flux_full(rn_full, fraction_h_full=0.95):
    return rn_full * fraction_h_full


def wind_speed_blending_height_bare(u_i, z0m_bare=0.001,
                                    z_obs=10, z_b=100):
    ws = ((CONST['k'] * u_i) / np.log(z_obs / z0m_bare)
          * np.log(z_b / z0m_bare) / CONST['k'])
    ws = np.where(np.isfinite(ws), ws, np.nan)
    ws_clip = np.clip(ws, 1, 150)

    return ws_clip


def wind_speed_blending_height_full_inst(u_i, z0m_full=0.1,
                                         z_obs=10, z_b=100):
    ws = ((CONST['k'] * u_i) / np.log(z_obs / z0m_full) *
          np.log(z_b / z0m_full) / CONST['k'])
    ws = np.where(np.isfinite(ws), ws, np.nan)
    ws_clip = np.clip(ws, 1, 150)

    return ws_clip


def friction_velocity_full_inst(u_b_i_full, z0m_full=0.1,
                                disp_full=0.667, z_b=100):
    return initial_friction_velocity_inst(u_b_i_full,
                                          z0m_full,
                                          disp_full,
                                          z_b=100)


def friction_velocity_bare_inst(u_b_i_bare, z0m_bare=0.001,
                                disp_bare=0.0, z_b=100):
    return initial_friction_velocity_inst(u_b_i_bare,
                                          z0m_bare, disp_bare,
                                          z_b=100)


def monin_obukhov_length_bare(h_bare, ad_i, u_star_i_bare,
                              t_air_k_i):
    return monin_obukhov_length(h_bare, ad_i, u_star_i_bare,
                                t_air_k_i)


def monin_obukhov_length_full(h_full, ad_i, u_star_i_full,
                              t_air_k_i):
    return monin_obukhov_length(h_full, ad_i, u_star_i_full,
                                t_air_k_i)


def aerodynamical_resistance_full(u_i, L_full, z0m_full=0.1,
                                  disp_full=0.667, z_obs=10):
    z1 = ne.evaluate("(z_obs - disp_full) / z0m_full")
    z2 = ne.evaluate("(z_obs - disp_full) / L_full")
    z3 = ne.evaluate("z0m_full / L_full")
    z4 = ne.evaluate("(z_obs - disp_full) / (z0m_full / 7)")
    z5 = ne.evaluate("(z0m_full / 7) / L_full")
    k = CONST['k']

    psim2 = psi_m(-z2)
    psim3 = psi_m(-z3)
    psih2 = psi_h(-z2)
    psih5 = psi_h(-z5)

    res = ne.evaluate("((log(z1) - psim2 + psim3) * "
                      "(log(z4) - psih2 + psih5))/(k ** 2 * u_i)")
    res = np.where(np.isfinite(res), res, np.nan)
    return res


def aerodynamical_resistance_bare(u_i, L_bare, z0m_bare=0.001,
                                  disp_bare=0.0, z_obs=10):
    z1 = ne.evaluate("(z_obs - disp_bare) / z0m_bare")
    z2 = ne.evaluate("(z_obs - disp_bare) / L_bare")

    psim2 = psi_m(-z2)
    psih2 = psi_h(-z2)
    k = CONST['k']

    res = ne.evaluate("((log(z1) - psim2) * (log(z1) - psih2)) "
                      "/ (k ** 2 * u_i)")
    res = np.where(np.isfinite(res), res, np.nan)

    return res


def wind_speed_soil_inst(u_i, L_bare, z_obs=10):
    z0_soil = 0.01
    z0_free = 0.1
    psimz = psi_m(-z0_free/L_bare)

    res = ne.evaluate("u_i * ((log(z0_free / z0_soil))/"
                      "(log(z_obs / z0_soil) - psimz))")
    res = np.where(np.isfinite(res), res, np.nan)

    return res


def aerodynamical_resistance_soil(u_i_soil):
    Tdif = 10.0
    return ne.evaluate("1. / (0.0025 * (Tdif) ** (1. / 3.) + "
                       "0.012 * u_i_soil)")


def maximum_temperature_full(ra_hor_clear_i,
                             emiss_atm_i, t_air_k_i,
                             ad_i, rac, r0_full=0.18):
    emiss_full = 0.99
    sb = CONST['sb']
    sh = CONST['sh']
    tc_max_num = ne.evaluate(
        "(1 - r0_full) * ra_hor_clear_i + emiss_full "
        "* emiss_atm_i * sb * (t_air_k_i) ** 4- emiss_full "
        "* sb * (t_air_k_i) ** 4")
    tc_max_denom = ne.evaluate("4 * emiss_full * sb * "
                               "(t_air_k_i) ** 3 + "
                               "(ad_i * sh) / rac")
    tc_max = ne.evaluate("tc_max_num / tc_max_denom + t_air_k_i")
    return tc_max


def maximum_temperature_bare(ra_hor_clear_i, emiss_atm_i,
                             t_air_k_i, ad_i, raa, ras,
                             r0_bare=0.38):
    emiss_bare = 0.95
    sb = CONST['sb']
    sh = CONST['sh']
    ts_max_num = ne.evaluate(
        "(1 - r0_bare) * ra_hor_clear_i + emiss_bare *"
        "emiss_atm_i * sb * (t_air_k_i) ** 4 - emiss_bare"
        "* sb * (t_air_k_i) ** 4")
    ts_max_denom = ne.evaluate("4 * emiss_bare * sb * "
                               "(t_air_k_i) ** 3 + "
                               "(ad_i * sh) / ((raa + ras) "
                               "* (1 - 0.35))")

    return ne.evaluate("ts_max_num / ts_max_denom + t_air_k_i")


def maximum_temperature(t_max_bare, t_max_full, vc):
    return vc * (t_max_full - t_max_bare) + t_max_bare


def minimum_temperature(t_wet_k_i, t_air_k_i, vc):
    return vc * (t_air_k_i - t_wet_k_i) + t_wet_k_i


def soil_moisture_from_maximum_temperature(lst_max, lst, lst_min):
    ratio = ne.evaluate("(lst - lst_min) / (lst_max - lst_min)")
    ratio[ratio < 0] = 0
    ratio[ratio > 1] = 1

    return 1 - ratio


def corrected_temperature(se_root, lst_lower=290, lst_upper=310):
    return (1 - se_root) * (lst_upper - lst_lower) + lst_lower


def soil_moisture(ndvi, l8ts, boundaries):
    ndvi_min = np.nanquantile(ndvi, 0.05, axis=0)
    ndvi_max = np.nanquantile(ndvi, 0.95, axis=0)
    Fc = (ndvi - ndvi_min) / (ndvi_max - ndvi_min)
    Fc = np.where(Fc < 0, 0, Fc)
    Fc = np.where(Fc > 1, 1, Fc)

    Ts_max = boundaries.select_bands(['Ts_max']).data[0]
    Tc_max = boundaries.select_bands(['Tc_max']).data[0]
    Ts_min = boundaries.select_bands(['Ts_min']).data[0]
    Tc_min = boundaries.select_bands(['Tc_min']).data[0]

    T_dry = abs(Ts_max-Tc_max)*Fc + Tc_max + 10
    T_wet = abs(Ts_min-Tc_min)*Fc + Ts_min

    b10 = l8ts.data[0]
    sm = (T_dry-(b10))/(T_dry-T_wet)
    sm = np.where(sm > 1, 1, sm)
    sm = np.where(sm < 0, 0, sm)
    # Fully saturated soil at negative temperatures
    sm = np.where(b10 < CONST['zero_celcius'], 1, sm)

    sm_stress = stress_moisture(sm, tenacity=1.5)

    sm = Timeseries(np.expand_dims(sm, axis=0),
                    l8ts.timestamps, ['sm'])
    sm_stress = Timeseries(np.expand_dims(sm_stress, axis=0),
                           l8ts.timestamps, ['smstress'])

    return sm, sm_stress


def stress_moisture(se_root, tenacity=1.5):
    stress = (tenacity * se_root
              - (np.sin(2 * math.pi * se_root))
              / (2 * math.pi))
    return np.clip(stress, 0, 1)


def theoretical_boundaries(agera5ts):
    alb_s = 0.38
    alb_c = 0.18

    e_s = 0.95
    e_c = 0.98
    p = 1.225
    Cp = CONST['sh']
    G_Rn = 0.35
    sigma = 5.67 * 10**-8

    t_dew_k_i = agera5ts.select_bands(['dewpoint_temperature']).data[0]
    t_air_k_i = agera5ts.select_bands(['temperature_mean']).data[0]
    u_i = agera5ts.select_bands(['wind_speed']).data[0]
    vp_i = agera5ts.select_bands(['vapour_pressure']).data[0]
    ra_hor_clear_i = agera5ts.select_bands(['solar_radiation_flux']).data[0]

    lst_bare = t_air_k_i + 30
    lst_full = t_air_k_i + 10

    Sd = ra_hor_clear_i * 1.15740741 * 10**-5

    # Compute wet and dry boundaries
    emiss_atm_i = atmospheric_emissivity_inst(vp_i, t_air_k_i)
    rn_bare = net_radiation_bare(ra_hor_clear_i, emiss_atm_i,
                                 t_air_k_i, lst_bare)
    h_bare = sensible_heat_flux_bare(rn_bare)
    ad_i = Cp
    u_b_i_bare = wind_speed_blending_height_bare(u_i)
    u_star_i_bare = friction_velocity_bare_inst(u_b_i_bare)
    L_bare = monin_obukhov_length_bare(h_bare, ad_i, u_star_i_bare,
                                       t_air_k_i)
    raa = aerodynamical_resistance_bare(u_i, L_bare)

    e_a = atmospheric_emissivity_inst(vp_i, t_air_k_i)

    u_i_soil = wind_speed_soil_inst(u_i, L_bare)
    ras = aerodynamical_resistance_soil(u_i_soil)

    u_b_i_full = wind_speed_blending_height_full_inst(u_i)
    u_star_i_full = friction_velocity_full_inst(u_b_i_full)
    rn_full = net_radiation_full(ra_hor_clear_i, emiss_atm_i,
                                 t_air_k_i, lst_full)
    h_full = sensible_heat_flux_full(rn_full)
    L_full = monin_obukhov_length_full(h_full, ad_i,
                                       u_star_i_full, t_air_k_i)
    rac = aerodynamical_resistance_full(u_i, L_full)

    Ts_max = (((1 - alb_s) * Sd + e_s * e_a * sigma * t_air_k_i**4
               - e_s * sigma * t_air_k_i**4) /
              (4 * e_s * sigma * t_air_k_i**3 + p * Cp /
               ((raa + ras) * (1 - G_Rn))) + t_air_k_i)
    Ts_max = np.where(Ts_max > 340, t_air_k_i, Ts_max + 10)

    Tc_max = (((1 - alb_c) * Sd + e_c * e_a * sigma * t_air_k_i**4
               - e_c * sigma * t_air_k_i**4) /
              (4 * e_c * sigma * t_air_k_i**3 + p * Cp /
               rac) + t_air_k_i)
    Tc_max = np.where(Tc_max > 340, t_air_k_i, Tc_max + 10)

    t_wb_k_i = wet_bulb_temperature_inst(t_air_k_i, t_dew_k_i)
    Ts_min = t_wb_k_i
    Tc_min = t_air_k_i

    # Set wet boundary at freezing point
    Ts_max = np.where(Ts_max < CONST['zero_celcius'],
                      CONST['zero_celcius'], Ts_max)
    Tc_max = np.where(Tc_max < CONST['zero_celcius'],
                      CONST['zero_celcius'], Tc_max)
    Ts_min = np.where(Ts_min < CONST['zero_celcius'],
                      CONST['zero_celcius'], Ts_min)
    Tc_min = np.where(Tc_min < CONST['zero_celcius'],
                      CONST['zero_celcius'], Tc_min)

    Ts_max = np.where(np.isfinite(Ts_max), Ts_max, np.nan)
    Tc_max = np.where(np.isfinite(Tc_max), Tc_max, np.nan)
    Ts_min = np.where(np.isfinite(Ts_min), Ts_min, np.nan)
    Tc_min = np.where(np.isfinite(Tc_min), Tc_min, np.nan)

    data = np.stack([Ts_max, Tc_max, Ts_min, Tc_min])
    names = ['Ts_max', 'Tc_max', 'Ts_min', 'Tc_min']

    return Timeseries(data, agera5ts.timestamps, names)


###################################################################
# alternative way of computing surface soil moisture
# (purely based on Sentinel-2)
###################################################################

@numba.njit
def coefficients(ndvi_range, edge):
    """
    Function to derive slope and intercept of the dry/wet edge,
    using the STR observations
    for wet and dry conditions for each discrete ndvi step. 
    """
    X = np.array(ndvi_range)
    y = np.array(edge)
    if np.isnan(y[0]):
        X = X[1:]
        y = y[1:]
    mean_x = np.mean(X)
    mean_y = np.mean(y)
    m = len(X)
    numer = 0
    denom = 0
    for i in range(m):
        numer += (X[i] - mean_x)*(y[i] - mean_y)
        denom += (X[i] - mean_x)**2
    slope = numer/denom
    intercept = mean_y - (slope * mean_x)
    return slope, intercept


@numba.njit
def get_SSM(x):
    """
    Function to calculate the surface soil moisture content
    (according to the OPTRAM model)
    using ndvi and surface transformed reflection (STR) timeseries.
    The ndvi timeseries is seperated into descrete steps which are 
    defined by the standard deviation of the whole ndvi timeseries.
    Within these steps, the range in STR is calculated and stored as
    the dry and wet edge of that specific ndvi range.
    Once for every ndvi step the wet and dry values of the STR are stored,
    the actual wet and dry edges for the optical trapezoid are defined
    using a linear regression analysis of the wet and dry points
    derived per ndvi step.
    Based on these edges the soil moisture content is calculated. 
    """
    ndvi = x[0]
    STR = x[1]
    max_ndvi = np.nanmax(ndvi)
    if max_ndvi < 0.5:
        min_ndvi = 0
    else:
        min_ndvi = 25
    ndvi_step = int((np.nanstd(ndvi)/3)*100)
    if ndvi_step < 1:
        ndvi_step = 1
    ndvi_range = [0]
    dry_list = [0]
    wet_list = [np.nan]
    for i in numba.prange(min_ndvi, 100, ndvi_step):  # pylint: disable=not-an-iterable
        i_min = i/100
        i_max = (i+ndvi_step)/100
        i_mean = (i_min+i_max)/2
        mask = np.where((ndvi >= i_min) & (ndvi < i_max), 1, np.nan)
        if not np.nansum(mask) > 0:
            continue
        STR_masked = STR*mask
        STR_masked = STR_masked[np.isfinite(STR_masked)]
        if not np.nansum(STR_masked) > 0:
            continue
        dry = np.nanmin(STR_masked)
        wet = np.nanmedian(STR_masked) + np.nanstd(STR_masked)
        if (not np.isnan(dry)) | (not np.isnan(wet)):
            ndvi_range.append(i_mean)
            dry_list.append(dry)
            wet_list.append(wet)
    if len(ndvi_range) < 3:
        i_d = np.nan
        i_w = np.nan
        s_d = np.nan
        s_w = np.nan
    else:
        s_d, i_d = coefficients(ndvi_range, dry_list)
        s_w, i_w = coefficients(ndvi_range, wet_list)
    SSM = (i_d + (s_d*ndvi) - STR)/(i_d - i_w + (s_d-s_w)*ndvi)
    SSM = np.where(SSM > 1, 1, SSM)
    SSM = np.where(SSM < 0, 0, SSM)
    return SSM


@numba.njit
def correlate_P_SSM(x):
    """
    Function to correlate the modelled SSM with the AgERA5 precipitation. 
    A high correlation indicates that the soil moisture is driven by
    precipitation. 
    A low correlation shows that other factors, like irrigation,
    could have caused the high soil moisture observations. 
    """
    X_ = x[0]
    y_ = x[1]
    N = len(X_)
    X_idx = np.argwhere(np.isfinite(X_)).flatten()
    y_idx = np.argwhere(np.isfinite(y_)).flatten()
    if not np.array_equal(X_idx, y_idx):
        if len(X_idx) > len(y_idx):
            X = np.full((len(y_idx)), 0., X_.dtype)
            y = np.full((len(y_idx)), 0., y_.dtype)
            for i, idx in zip(range(0, len(y)), y_idx):
                X[i] = X_[idx]
                y[i] = y_[idx]
        else:
            X = np.full((len(X_idx)), 0., X_.dtype)
            y = np.full((len(X_idx)), 0., y_.dtype)
            for i, idx in zip(range(0, len(X)), X_idx):
                X[i] = X_[idx]
                y[i] = y_[idx]
    else:
        X = np.copy(X_)
        y = np.copy(y_)
    mean_x = np.nanmean(X)
    mean_y = np.nanmean(y)
    m = len(X)
    numer = 0
    denom = 0
    for i in range(m):
        numer += (X[i] - mean_x)*(y[i] - mean_y)
        denom += (X[i] - mean_x)**2
    if denom == 0:
        m = 0
    else:
        m = numer/denom
    c = mean_y - (m * mean_x)
    ss_t = 0
    ss_r = 0
    y_pred_list = []
    for i in range(len(X)):
        y_pred = c + m*X[i]
        y_pred_list.append(y_pred)
        ss_t += (y[i] - mean_y)**2
        ss_r += (y[i] - y_pred)**2
    if ss_t == 0:
        r2 = 0
    else:
        r2 = 1 - (ss_r/ss_t)
    return np.array([r2]*N)


def feats2stack(par1, par2):
    par1 = to_stack(par1)
    par2 = to_stack(par2)
    return np.array([np.array(
        [par1_, par2_]) for par1_, par2_ in zip(par1, par2)])


def surface_soil_moisture(ndvi, str_1, mndwi,
                          timestamps,
                          precip):

    # compute final STR index
    str_2 = np.where(mndwi > 0.42, np.nan, str_1)

    # Calculate soil moisture
    SSM_func = np.vectorize(get_SSM, signature='(i,j)->(a)')
    SSM_stack = feats2stack(ndvi, str_2)
    ssm = SSM_func(SSM_stack)
    ssm = ssm.T.reshape(ndvi.shape[0],
                        ndvi.shape[1],
                        ndvi.shape[2])
    ssm = ssm.astype(np.float32)
    # set minimum ssm at 0.1
    ssm[ssm < 0.1] = 0.1

    # Calculate correlation between soil moisture and precipitation
    reg_func = np.vectorize(correlate_P_SSM, signature='(i,j)->(a)')
    reg_stack = feats2stack(precip, ssm)
    r2_P_SSM = reg_func(reg_stack)
    r2_P_SSM = r2_P_SSM.T.reshape(precip.shape[0],
                                  precip.shape[1],
                                  precip.shape[2])

    # Calculate precipitation adjusted SSM
    ssm_adj = ssm - r2_P_SSM*2

    # convert to timeseries objects
    ssm = Timeseries(np.expand_dims(ssm, axis=0),
                     timestamps, ['ssm'])
    ssm_adj = Timeseries(np.expand_dims(ssm_adj, axis=0),
                         timestamps, ['ssm_adj'])

    return ssm, ssm_adj
