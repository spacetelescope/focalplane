import copy

from astropy.table import Table
import astropy.units as u
from astropy.time import Time
import numpy as np

from pyia.data import GaiaData
from pystrometry import pystrometry


def correct_for_proper_motion(gaia_table, target_epoch, verbose=False, ignore_parallax=True):
    """Apply proper motion correction to an input Gaia catalog.

    Compute positions and uncertainties at an epoch other than the catalog epoch.

    Supports only Gaia input catalog format, i.e. and astropy table with Gaia-named columns.

    TODO:
    -----
        Do corrected_values['ra_error'] need to be corrected for cos(delta) effect?


    Parameters
    ----------
    gaia_table
    target_epoch : astropy time
    verbose
    ignore_parallax : bool
        If True, set parallax to zero to ignore its contribution to the offset
        (that offset is observer-dependent)

    Returns
    -------

    """
    gaia_table = copy.deepcopy(gaia_table)

    DR2_REF_EPOCH = gaia_table['ref_epoch'][0]

    for attribute_name in 'ra dec ra_error dec_error'.split():
        gaia_table[
            '{}_original_{}'.format(attribute_name, DR2_REF_EPOCH)] = np.full(
            len(gaia_table), np.nan)
        gaia_table['{}_{:3.1f}'.format(attribute_name, target_epoch.jyear)] = np.full(
            len(gaia_table), np.nan)


    gaia_data = GaiaData(gaia_table)

    for i in range(len(gaia_table)):
        if (not np.isnan(gaia_table['parallax'][i])) and (not np.ma.is_masked(gaia_table['parallax'][i])):
            gaia_star = gaia_data[i]
            covariance_matrix_mas = gaia_star.get_cov(units=dict(ra=u.milliarcsecond,
                                                                 dec=u.milliarcsecond,
                                                                 parallax=u.milliarcsecond,
                                                                 pm_ra=u.milliarcsecond/u.year,
                                                                 pm_dec=u.milliarcsecond/u.year))

            # remove radial velocity component
            covariance_matrix_mas = np.squeeze(covariance_matrix_mas)[0:5, 0:5]

            if verbose:
                print(covariance_matrix_mas)
                print(np.diag(covariance_matrix_mas))
                tbl_names = ['ra', 'dec', 'parallax', 'pmra', 'pmdec']
                for colname in tbl_names:
                    print('{} = {}'.format(colname, getattr(gaia_star, colname)))
                    err_colname = '{}_error'.format(colname)
                    print('{} = {}'.format(err_colname, getattr(gaia_star, err_colname)))

            # helper object to get PPM coefficients
            T = Table()
            T['MJD'] = [target_epoch.utc.mjd]
            T['frame'] = 1
            T['OB'] = 1
            iad = pystrometry.ImagingAstrometryData(T)
            iad.RA_deg = gaia_star.ra.to(u.deg).value
            iad.Dec_deg = gaia_star.dec.to(u.deg).value

            # this step depends on the observer when computing parallax factors
            # set reference epoch properly
            # https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
            # ref_epoch : Reference epoch (double, Time[Julian Years])
            # Reference epoch to which the astrometric source parameters are referred, expressed as a Julian Year in TCB.
            # At DR2 this reference epoch is always J2015.5 but in future releases this will be different and not necessarily the same for all sources.
            iad.set_five_parameter_coefficients(verbose=False, overwrite=False,
                                                reference_epoch_MJD=Time(gaia_star.ref_epoch[0], format='jyear', scale='tcb').utc.mjd)
            if verbose:
                print(iad.five_parameter_coefficients_table)
                print(iad.five_parameter_coefficients_array)

            if ignore_parallax:
                gaia_star.parallax = 0. * u.arcsec

            delta_ppm_array = np.array([0., 0.,
                                  gaia_star.parallax.to(u.deg).value[0],
                                  gaia_star.pmra.to(u.deg/u.year).value[0],
                                  gaia_star.pmdec.to(u.deg/u.year).value[0]])
            [delta_rastar_at_epoch_deg, delta_dec_at_epoch_deg] = np.dot(iad.five_parameter_coefficients_array.T, delta_ppm_array)
            dec_at_epoch_deg = gaia_star.dec.to(u.deg).value + delta_dec_at_epoch_deg
            if 0:
                cos_delta_factor = np.cos(np.deg2rad(gaia_star.dec.to(u.deg).value))
            else:
                # this is the way simbad is doing it
                cos_delta_factor = np.cos(np.deg2rad(dec_at_epoch_deg))
            ra_at_epoch_deg = gaia_star.ra.to(u.deg).value + delta_rastar_at_epoch_deg/cos_delta_factor

            corrected_values = {}
            for ii, jj in enumerate(iad.observing_1D_xi):
                prediction_vector = iad.five_parameter_coefficients_array.T[jj]
                prediction_uncertainty_x = np.sqrt(
                    np.dot(np.dot(prediction_vector, covariance_matrix_mas), prediction_vector))
                prediction_vector_y = iad.five_parameter_coefficients_array.T[jj + 1]
                prediction_uncertainty_y = np.sqrt(
                    np.dot(np.dot(prediction_vector_y, covariance_matrix_mas), prediction_vector_y))
                if verbose:
                    print(
                    '{}: (COV) offset and uncertainty in RA : {:3.12f} +/- {:3.12f} mas '.format(
                        target_epoch.utc.isot, ra_at_epoch_deg, prediction_uncertainty_x))
                    print(
                    '{}: (COV) offset and uncertainty in Dec: {:3.12f} +/- {:3.12f} mas '.format(
                        target_epoch.utc.isot, dec_at_epoch_deg, prediction_uncertainty_y))
                corrected_values['ra'] = ra_at_epoch_deg
                corrected_values['dec'] = dec_at_epoch_deg
                corrected_values['ra_error'] = prediction_uncertainty_x
                corrected_values['dec_error'] = prediction_uncertainty_y

            for attribute_name in 'ra dec ra_error dec_error'.split():
                gaia_table['{}_original_{}'.format(attribute_name, gaia_star.ref_epoch[0].value)][i] = \
                gaia_table[attribute_name][i]
                gaia_table['{}_{:3.1f}'.format(attribute_name, target_epoch.utc.jyear)][i] = \
                corrected_values[attribute_name]
                gaia_table['{}'.format(attribute_name)][i] = \
                gaia_table['{}_{:3.1f}'.format(attribute_name, target_epoch.utc.jyear)][i]
                if verbose:
                    print(
                    'Replacing {}={} by proper motion and parallax corrected value of {}'.format(
                        attribute_name,
                        gaia_table['{}_{}'.format(attribute_name, gaia_star.ref_epoch[0].value)][i],
                        gaia_table['{}'.format(attribute_name)][i]))

    return gaia_table

