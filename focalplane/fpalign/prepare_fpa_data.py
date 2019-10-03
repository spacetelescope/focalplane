import copy
from collections import OrderedDict
import os
import glob
import numpy as np
import pickle
import pylab as pl
import sys
import warnings

import astropy.io.fits as fits
from astropy.table import Table, vstack
import astroquery.mast as mast
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder, CircularAperture, RectangularAperture
from astropy.visualization import SqrtStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from jwst import datamodels

import pysiaf

from pystortion import crossmatch

from .alignment import AlignmentObservation, compute_idl_to_tel_in_table



try:
    # sys.path.append(os.path.join(home_dir, 'astro/code/packages/astrohelpers'))
    import gaia_helpers
except ImportError:
    print('Module gaia_helpers not available')

from scipy.spatial import cKDTree


def select_isolated_sources(extracted_sources, nearest_neighbour_distance_threshold_pix):
    """
    select isolated stars
    https://stackoverflow.com/questions/57129448/find-distance-to-nearest-neighbor-in-2d-array

    Parameters
    ----------
    extracted_sources
    nearest_neighbour_distance_threshold_pix

    Returns
    -------

    """
    stars_xy = np.array([extracted_sources['xcentroid'], extracted_sources['ycentroid']]).T
    tree = cKDTree(stars_xy)
    dists = tree.query(stars_xy, 2)
    nearest_neighbour_distance = dists[0][:, 1]

    extracted_sources.remove_rows(
        np.where(nearest_neighbour_distance < nearest_neighbour_distance_threshold_pix)[0])

    return extracted_sources

def crossmatch_fpa_data(parameters):
    """

    Parameters
    ----------
    parameters

    Returns
    -------

    """
    print('\nCROSSMATCH OF FPA DATA WITH REFERENCE CATALOG')
    if (not os.path.isfile(parameters['pickle_file']) or parameters['overwrite']):

        fpa_data_files = glob.glob(os.path.join(parameters['standardized_data_dir'], '*.fits'))
        verbose_figures = parameters['verbose_figures']
        save_plot = parameters['save_plot']
        plot_dir = parameters['plot_dir']
        out_dir = parameters['out_dir']
        q_max_cutoff = parameters['q_max_cutoff']
        siaf = parameters['siaf']
        gaia_tag = parameters['gaia_tag']
        verbose = parameters['verbose']
        idl_tel_method = parameters['idl_tel_method']
        reference_catalog = parameters['reference_catalog']
        xmatch_radius_camera = parameters['xmatch_radius_camera']
        xmatch_radius_fgs = parameters['xmatch_radius_fgs']
        rejection_level_sigma = parameters['rejection_level_sigma']
        restrict_analysis_to_these_apertures = parameters['restrict_analysis_to_these_apertures']

        observations = []
        for j, f in enumerate(fpa_data_files):

            print('=' * 40)
            fpa_data = Table.read(f)
            print('Read {} rows from {}'.format(len(fpa_data), f))

            if parameters['observatory'] == 'JWST':
                pl.close('all')
                print('Loading FPA observations in %s' % f)
                fpa_name_seed = os.path.basename(f).split('.')[0]

                aperture_name = fpa_data.meta['SIAFAPER']

                if (restrict_analysis_to_these_apertures is not None):
                    if (aperture_name not in restrict_analysis_to_these_apertures):
                        continue

                aperture = copy.deepcopy(siaf[aperture_name])
                reference_aperture = copy.deepcopy(aperture)

                print('using aperture: %s %s %s' % (aperture.observatory, aperture.InstrName, aperture.AperName))

                # compute v2 v3 coordinates of gaia catalog stars using reference aperture (using boresight)
                attitude_ref = pysiaf.utils.rotations.attitude(0., 0., fpa_data.meta['pointing_ra_v1'],
                                                               fpa_data.meta['pointing_dec_v1'],
                                                               fpa_data.meta['pointing_pa_v3'])

                reference_catalog['v2_spherical_arcsec'], reference_catalog[
                    'v3_spherical_arcsec'] = pysiaf.utils.rotations.getv2v3(attitude_ref, np.array(
                    reference_catalog['ra']), np.array(reference_catalog['dec']))

                # reference_catalog['v2_spherical_arcsec'][reference_catalog['v2_spherical_arcsec']>180*3600] -= 360*3600

                reference_cat = SkyCoord(
                    ra=np.array(reference_catalog['v2_spherical_arcsec']) * u.arcsec,
                    dec=np.array(reference_catalog['v3_spherical_arcsec']) * u.arcsec)

                # generate alignment observation
                obs = AlignmentObservation(aperture.observatory, aperture.InstrName)
                obs.aperture = aperture

                star_catalog = fpa_data

                star_catalog['star_id'] = star_catalog['id']
                obs.star_catalog = star_catalog

                # SCI science frame (in pixels) -> IDL frame (in arcsec)
                obs.star_catalog['x_idl_arcsec'], obs.star_catalog['y_idl_arcsec'] = aperture.sci_to_idl(np.array(obs.star_catalog['x_SCI']), np.array(obs.star_catalog['y_SCI']))

                # compute V2/V3
                # IDL frame in degrees ->  V2/V3_tangent_plane in arcsec
                # obs.compute_v2v3(aperture, method=idl_tel_method, input_coordinates='tangent_plane')
                obs.star_catalog = compute_idl_to_tel_in_table(obs.star_catalog, aperture, method=idl_tel_method)

                # define Gaia catalog specific to every aperture to allow for local tangent-plane projection
                v2v3_reference = SkyCoord(ra=reference_aperture.V2Ref * u.arcsec,
                                          dec=reference_aperture.V3Ref * u.arcsec)
                selection_index = np.where(reference_cat.separation(v2v3_reference) < 3 * u.arcmin)[0]

                # pl.figure()
                # pl.plot(reference_cat.ra, reference_cat.dec, 'b.')
                # pl.plot(v2v3_reference.ra, v2v3_reference.dec, 'ro')
                # pl.show()
                # 1/0

                obs.gaia_catalog = reference_catalog[selection_index]
                # obs.gaia_reference_catalog = reference_catalog


                # determine which Gaia stars fall into the aperture
                path_Tel = aperture.path('tel')
                mask = path_Tel.contains_points(np.array(
                    obs.gaia_catalog['v2_spherical_arcsec', 'v3_spherical_arcsec'].to_pandas()))

                # if verbose_figures:
                if 0:
                # if aperture.InstrName == 'NIRISS':
                    pl.figure()
                    pl.plot(obs.star_catalog['v2_spherical_arcsec'],
                            obs.star_catalog['v3_spherical_arcsec'], 'ko', mfc='w', mew=1)
                    pl.plot(obs.gaia_catalog['v2_spherical_arcsec'],
                            obs.gaia_catalog['v3_spherical_arcsec'], 'b.')
                    # pl.plot(reference_catalog['v2_spherical_arcsec'], reference_catalog['v3_spherical_arcsec'], 'k.')
                    pl.axis('equal')
                    pl.title(fpa_name_seed)
                    aperture.plot()
                    pl.show()
                    1/0
                    # aperture_names = ['FGS1', 'FGS2', 'FGS3']
                    # instruments = aperture_names
                    # for j, instrument in enumerate(instruments):
                    #     a = siaf.apertures[instrument]
                    #     a.plot(color='b')

                    ax = pl.gca()
                    # ax.invert_yaxis()
                    if save_plot == 1:
                        figure_name = os.path.join(plot_dir, '%s_v2v3_spherical.pdf' % fpa_name_seed)
                        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

                    1/0

                remove_multiple_matches = True
                retain_best_match = True

                # crossmatch star_catalog with gaia_catalog
                star_cat = SkyCoord(ra=np.array(obs.star_catalog['v2_spherical_arcsec']) * u.arcsec,
                                    dec=np.array(
                                        obs.star_catalog['v3_spherical_arcsec']) * u.arcsec)

                gaia_cat = SkyCoord(ra=np.array(obs.gaia_catalog['v2_spherical_arcsec']) * u.arcsec,
                                    dec=np.array(
                                        obs.gaia_catalog['v3_spherical_arcsec']) * u.arcsec)

                # # tackle wrapping or RA coordinates
                # if np.ptp(star_cat.ra).value > 350:
                #     star_cat.ra[np.where(star_cat.ra > 180 * u.deg)[0]] -= 360 * u.deg
                # if np.ptp(gaia_cat.ra).value > 350:
                #     gaia_cat.ra[np.where(gaia_cat.ra > 180 * u.deg)[0]] -= 360 * u.deg

                xmatch_radius = copy.deepcopy(xmatch_radius_camera)

                idx_gaia_cat, idx_star_cat, d2d, d3d, diff_raStar, diff_de = crossmatch.xmatch(
                    gaia_cat, star_cat, xmatch_radius, rejection_level_sigma, verbose=verbose,
                    verbose_figures=verbose_figures, saveplot=save_plot, out_dir=plot_dir,
                    name_seed=fpa_name_seed, retain_best_match=retain_best_match,
                    remove_multiple_matches=remove_multiple_matches)

                print(
                    '{:d} measured stars, {:d} reference catalog stars in the aperture, {:d} matches.'.format(
                        len(obs.star_catalog), np.sum(mask), len(idx_gaia_cat)))
                # 1/0

            elif parameters['observatory'] == 'HST':


                for camera_name in parameters['camera_names']:
                    if fpa_data.meta['INSTRUME'] != camera_name:
                        continue
                    else:
                        pl.close('all')
                        print('Loading FPA observations in %s' % f)

                    fpa_name_seed = os.path.basename(f).split('.')[0]

                    if 'FGS' not in camera_name:
                        retain_best_match = 1

                        # make and save q figure
                        if verbose_figures:
                            fig = pl.figure(figsize=(7, 7), facecolor='w', edgecolor='k')
                            pl.clf()
                            pl.plot(fpa_data['m'], fpa_data['q'], 'bo')
                            pl.plot(fpa_data['m'][fpa_data['q'] > q_max_cutoff],
                                    fpa_data['q'][fpa_data['q'] > q_max_cutoff], 'ro', mfc='w')
                            pl.title(
                                '{}: {}'.format(fpa_data.meta['INSTRUME'], fpa_data.meta['DATAFILE']))
                            pl.xlabel('m')
                            pl.ylabel('q')
                            pl.axhline(q_max_cutoff, ls='--')
                            # pl.ylim((0,q_max_cutoff))
                            # pl.hist(fpa_data['q'], 50)
                            pl.show()
                            if save_plot == 1:
                                figure_name = os.path.join(plot_dir, '%s_onepass_q.pdf' % fpa_name_seed)
                                pl.savefig(figure_name, transparent=True, bbox_inches='tight',
                                           pad_inches=0)

                        fpa_data.remove_rows(np.where(fpa_data['q'] > q_max_cutoff)[0])

                        instrument = fpa_data.meta['INSTRUME']

                        image_file_name = os.path.join(fpa_data.meta['DATAPATH'],
                                                       fpa_data.meta['DATAFILE'])
                        primary_header = fits.getheader(image_file_name, ext=0)
                        # chip2_header = fits.getheader(image_file_name, ext=1)
                        # chip1_header = fits.getheader(image_file_name, ext=4)


                        header_aperture_name = fpa_data.meta['APERTURE'].strip()

                        # set SIAF reference aperture and aperture (aperture names in FITS header differ from SIAF names)
                        if header_aperture_name == 'WFC-FIX':
                            reference_aperture_name = 'JWFCFIX'
                            if fpa_data.meta['CHIP'] == 1:
                                aperture_name = 'JWFC1FIX'
                            elif fpa_data.meta['CHIP'] == 2:
                                aperture_name = 'JWFC2FIX'
                        elif header_aperture_name == 'UVIS-CENTER':
                            reference_aperture_name = 'IUVISCTR'
                            if fpa_data.meta['CHIP'] == 1:
                                aperture_name = 'IUVIS1FIX'
                            elif fpa_data.meta['CHIP'] == 2:
                                aperture_name = 'IUVIS2FIX'

                    else:  # FGS is in aperture name
                        header_aperture_name = fpa_data.meta['SIAFAPER']
                        reference_aperture_name = header_aperture_name
                        aperture_name = header_aperture_name

                        # if aperture_name == 'FGS1':
                        #     verbose_figures = True


                    if (restrict_analysis_to_these_apertures is not None):
                        if (aperture_name not in restrict_analysis_to_these_apertures):
                            continue

                    aperture = copy.deepcopy(siaf.apertures[aperture_name])
                    reference_aperture = siaf.apertures[reference_aperture_name]
                    print('using aperture: %s %s %s' % (
                    aperture.observatory, aperture.InstrName, aperture.AperName))
                    name_seed = fpa_name_seed

                    # compute v2 v3 coordinates of gaia catalog stars using reference aperture (using boresight)
                    attitude_ref = pysiaf.utils.rotations.attitude(0., 0.,
                                                                   fpa_data.meta['RA_V1'],
                                                                   fpa_data.meta['DEC_V1'],
                                                                   fpa_data.meta['PA_V3'])

                    reference_catalog['v2_spherical_arcsec'], reference_catalog[
                        'v3_spherical_arcsec'] = pysiaf.utils.rotations.getv2v3(attitude_ref, np.array(
                        reference_catalog['ra']), np.array(reference_catalog['dec']))

                    gaia_reference_cat = SkyCoord(
                        ra=np.array(reference_catalog['v2_spherical_arcsec']) * u.arcsec,
                        dec=np.array(reference_catalog['v3_spherical_arcsec']) * u.arcsec)

                    # make overview plot with all relevant apertures and gaia stars
                    if 0:
                        fig = pl.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
                        for aperture_name, aperture in siaf.apertures.items():
                            # aperture.plot(label=True, color='k')
                            show_label = True
                            aperture.plot(color='k')
                            if 'FGS' in aperture_name:
                                label = aperture_name
                            elif aperture_name == 'IUVISCTR':
                                label = 'WFC3'
                            elif aperture_name == 'JWFCFIX':
                                label = 'ACS'
                            else:
                                show_label = False

                            if show_label:
                                pl.text(aperture.V2Ref, aperture.V3Ref, label , horizontalalignment='center')

                        pl.plot(reference_catalog['v2_spherical_arcsec'],
                                reference_catalog['v3_spherical_arcsec'], 'k.', ms=1,
                                mfc='0.7')
                        # pl.axis('tight')
                        pl.axis('equal')
                        pl.xlim((-1000, 1000))
                        pl.ylim((-1000, 1000))
                        pl.xlabel('V2 (arcsec)')
                        pl.ylabel('V3 (arcsec)')
                        ax = pl.gca()
                        ax.invert_yaxis()
                        pl.show()
                        if save_plot == 1:
                            figure_name = os.path.join(plot_dir, '%s_overview_v2v3_with_gaia.pdf' % name_seed)
                            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)
                        1/0


                    # generate alignment observation
                    obs = AlignmentObservation(aperture.observatory, aperture.InstrName)
                    obs.aperture = aperture

                    star_catalog = fpa_data


                    if 'FGS' not in camera_name:
                        # assign star identifiers
                        star_catalog['star_id'] = star_catalog['n']
                        obs.star_catalog = star_catalog

                        if fpa_data.meta['INSTRUME'] == 'ACS':
                            number_of_reference_pixels_x = 24
                        elif fpa_data.meta['INSTRUME'] == 'WFC3':
                            number_of_reference_pixels_x = 25

                        # for HST the SIAF SciRef coordinates include the reference pixels. see also Colin's excel worksheet and his email dated September 22 2017
                        # SCI science frame (in pixels) -> IDL frame (in arcsec)
                        obs.star_catalog['x_idl_arcsec'], obs.star_catalog['y_idl_arcsec'] = aperture.sci_to_idl(
                            np.array(obs.star_catalog['x_SCI']) + number_of_reference_pixels_x,
                            np.array(obs.star_catalog['y_SCI']))

                    else:
                        # assign star identifiers
                        star_catalog['star_id'] = [np.int(s.replace('_', '')) for s in star_catalog['TARGET_ID']]
                        obs.star_catalog = star_catalog

                    # compute V2/V3
                    # IDL frame in degrees ->  V2/V3_tangent_plane in arcsec
                    # obs.compute_v2v3(aperture, method=idl_tel_method, input_coordinates='tangent_plane')
                    obs.star_catalog = compute_idl_to_tel_in_table(obs.star_catalog, aperture, method=idl_tel_method)

                    # define Gaia catalog specific to every aperture to allow for local tangent-plane projection
                    v2v3_reference = SkyCoord(ra=reference_aperture.V2Ref * u.arcsec, dec=reference_aperture.V3Ref * u.arcsec)
                    if 'FGS' not in camera_name:
                        gaia_selection_index = np.where(gaia_reference_cat.separation(v2v3_reference) < 3 * u.arcmin)[0]
                    else:
                        # some FGS stars were observed several times to correct for drifts etc.
                        tmp, tmp_unique_index = np.unique(obs.star_catalog['STAR_NAME'].data,
                                                          return_index=True)
                        tmp_star_cat = SkyCoord(
                            ra=np.array(obs.star_catalog['RA'][tmp_unique_index]) * u.deg,
                            dec=np.array(obs.star_catalog['DEC'][tmp_unique_index]) * u.deg)

                        tmp_gaia_cat = SkyCoord(ra=np.array(reference_catalog['ra']) * u.deg,
                                                dec=np.array(reference_catalog['dec']) * u.deg)

                        xmatch_radius_sky = 1 * u.arcsecond
                        rejection_level_sigma = 5
                        retain_best_match = 1
                        tmp_idx_gaia_cat, tmp_idx_star_cat, d2d, d3d, diff_raStar, diff_de = pystortion.crossmatch.xmatch(
                            tmp_gaia_cat, tmp_star_cat,
                            xmatch_radius_sky,
                            verbose=0, rejection_level_sigma=rejection_level_sigma,
                            retain_best_match=retain_best_match, verbose_figures=0)
                        gaia_selection_index = tmp_idx_gaia_cat
                        print('{} unique stars in the FGS observations'.format(len(tmp_unique_index)))
                        print('{} unique FGS stars crossmatched with Gaia (in sky frame)'.format(len(gaia_selection_index)))

                    obs.gaia_catalog = reference_catalog[gaia_selection_index]
                    obs.gaia_reference_catalog = reference_catalog

                    # IMPLEMENT PROPER MOTION CORRECTION HERE
                    target_epoch = Time(fpa_data.meta['EPOCH'], format='isot')

                    if parameters['correct_gaia_for_proper_motion']:
                        # plot_position_errors = True
                        file_format = '.fits'
                        output_name_seed = gaia_tag + '_{:.3f}'.format(target_epoch.decimalyear)
                        pm_corrected_file = os.path.join(out_dir,
                                                         'gaia_sources_{}_pm_corrected_{}'.format(
                                                             gaia_tag,
                                                             target_epoch.isot)) + file_format
                        if (parameters['overwrite_pm_correction']) or (not os.path.isfile(pm_corrected_file)):
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore', AstropyWarning, append=True)
                                warnings.simplefilter('ignore', UserWarning, append=True)
                                obs.gaia_catalog = gaia_helpers.correct_for_proper_motion(
                                    obs.gaia_catalog,
                                    target_epoch,
                                    verbose=False)
                            obs.gaia_catalog.write(pm_corrected_file, overwrite=True)
                        else:
                            obs.gaia_catalog = Table.read(pm_corrected_file)

                    # overflow_index = np.where(obs.star_catalog['v2_spherical_arcsec'] / 3600 > 350.)[0]
                    # obs.star_catalog['v2_spherical_arcsec'][overflow_index] -= 360. * 3600

                    # determine which Gaia stars fall into the aperture
                    path_Tel = aperture.path('tel')
                    mask = path_Tel.contains_points(np.array(obs.gaia_catalog['v2_spherical_arcsec', 'v3_spherical_arcsec'].to_pandas()))

                    if 'FGS' not in camera_name:
                        if verbose_figures:
                            pl.figure()
                            pl.plot(obs.star_catalog['v2_spherical_arcsec'],
                                    obs.star_catalog['v3_spherical_arcsec'], 'ko', mfc='w', mew=1)
                            pl.plot(obs.gaia_catalog['v2_spherical_arcsec'],
                                    obs.gaia_catalog['v3_spherical_arcsec'], 'b.')
                            pl.axis('equal')
                            pl.title(fpa_name_seed)
                            pl.show()

                        # plot the aperture contours in v2v3
                        if fpa_data.meta['INSTRUME'] == 'ACS':
                            aperture_names = [reference_aperture_name, 'JWFC1FIX', 'JWFC2FIX']
                        elif fpa_data.meta['INSTRUME'] == 'WFC3':
                            aperture_names = [reference_aperture_name, 'IUVIS1FIX', 'IUVIS2FIX']
                        instruments = [fpa_data.meta['INSTRUME']] * len(aperture_names)

                        for AperName in aperture_names:
                            siaf.apertures[AperName].plot('tel')  # , label=True)

                    else:
                        if verbose_figures:
                            pl.figure()
                            pl.plot(obs.star_catalog['v2_spherical_arcsec'],
                                    obs.star_catalog['v3_spherical_arcsec'], 'ko', mfc='w',
                                    mew=1)
                            pl.plot(obs.gaia_catalog['v2_spherical_arcsec'],
                                    obs.gaia_catalog['v3_spherical_arcsec'], 'b.')
                            pl.axis('equal')
                            pl.title(fpa_name_seed)
                            pl.show()

                            aperture_names = ['FGS1', 'FGS2', 'FGS3']
                            instruments = aperture_names
                            for j, instrument in enumerate(instruments):
                                a = siaf.apertures[instrument]
                                a.plot(color='b')

                        ax = pl.gca()
                        ax.invert_yaxis()
                        if save_plot == 1:
                            figure_name = os.path.join(plot_dir,
                                                       '%s_v2v3_spherical.pdf' % fpa_name_seed)
                            pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

                    if 'FGS' not in camera_name:
                        remove_multiple_matches = True
                    else:
                        # keep the FGS stars that were observed several times by the astrometer
                        remove_multiple_matches = False

                    # crossmatch star_catalog with gaia_catalog
                    star_cat = SkyCoord(ra=np.array(obs.star_catalog['v2_spherical_arcsec']) * u.arcsec,
                                        dec=np.array(obs.star_catalog['v3_spherical_arcsec']) * u.arcsec)

                    gaia_cat = SkyCoord(ra=np.array(obs.gaia_catalog['v2_spherical_arcsec']) * u.arcsec,
                                        dec=np.array(obs.gaia_catalog['v3_spherical_arcsec']) * u.arcsec)


                    # if aperture_name == 'FGS1':
                        # xmatch_radius = 1.*u.arcsec
                        # verbose_figures = True
                        # verbose=True
                        # 1/0
                    #     xmatch_radius = 5.*u.arcsec
                    # else:
                    if 'FGS' in aperture_name:
                        xmatch_radius = copy.deepcopy(xmatch_radius_fgs)
                    else:
                        xmatch_radius = copy.deepcopy(xmatch_radius_camera)

                    # run xmatch
                    if 0:
                        idx_gaia_cat, idx_star_cat, d2d, d3d, diff_raStar, diff_de = distortion.crossmatch_sky_catalogues_with_iterative_distortion_correction(
                            gaia_cat, star_cat, out_dir, verbose=1,
                            initial_crossmatch_radius=xmatch_radius, max_iterations=10,
                            n_interation_switch=1, k=6, overwrite=True, adaptive_xmatch_radius_factor=5)
                        1 / 0
                    else:
                        # verbose = 0
                        # verbose_figures = 0
                        idx_gaia_cat, idx_star_cat, d2d, d3d, diff_raStar, diff_de = \
                            pystortion.crossmatch.xmatch(gaia_cat, star_cat,
                                                         xmatch_radius,
                                                         rejection_level_sigma,
                                                         verbose=verbose,
                                                         verbose_figures=verbose_figures,
                                                         saveplot=save_plot,
                                                         out_dir=plot_dir,
                                                         name_seed=fpa_name_seed,
                                                         retain_best_match=retain_best_match,
                                                         remove_multiple_matches=remove_multiple_matches)

                    print(
                        '{:d} measured stars, {:d} Gaia catalog stars in the aperture, {:d} matched with Gaia.'.format(
                            len(obs.star_catalog), np.sum(mask), len(idx_gaia_cat)))

                    # if aperture_name == 'FGS1':
                    #     1/0

                    if ('FGS' in camera_name) & (len(obs.star_catalog) != len(idx_gaia_cat)):
                        print('MISSING STAR:')
                        obs.star_catalog[np.setdiff1d(np.arange(len(obs.star_catalog)), idx_star_cat)].pprint()

                        if 0:
                            pl.figure()
                            pl.plot(obs.star_catalog['v2_spherical_arcsec'],
                                    obs.star_catalog['v3_spherical_arcsec'], 'ko', mfc='w',
                                    mew=1)
                            pl.plot(obs.gaia_catalog['v2_spherical_arcsec'],
                                    obs.gaia_catalog['v3_spherical_arcsec'], 'b.')
                            pl.axis('equal')
                            pl.title(fpa_name_seed)
                            pl.show()

                            aperture_names = ['FGS1', 'FGS2', 'FGS3']
                            instruments = aperture_names  # [fpa_data.meta['INSTRUME']] * len(
                            # aperture_names)

                            for j, instrument in enumerate(instruments):
                                a = siaf.apertures[instrument]
                                a.plot(color='b')

                            ax = pl.gca()
                            ax.invert_yaxis()


            obs.number_of_measured_stars = len(obs.star_catalog)
            obs.number_of_gaia_stars = np.sum(mask)
            obs.number_of_matched_stars = len(idx_gaia_cat)

            obs.gaia_catalog_matched = obs.gaia_catalog[idx_gaia_cat]
            obs.star_catalog_matched = obs.star_catalog[idx_star_cat]
            obs.gaia_catalog_matched['star_id'] = obs.star_catalog_matched['star_id']

            # save space in pickle, speed up
            obs.gaia_catalog = []
            obs.star_catalog = []
            obs.gaia_reference_catalog = []

            obs.siaf_aperture_name = aperture_name
            obs.fpa_data = fpa_data
            obs.fpa_name_seed = fpa_name_seed

            # dictionary that defines the names of columns in the star/gaia_catalog for use later on
            fieldname_dict = {}
            fieldname_dict['star_catalog'] = {}  # observed
            fieldname_dict['reference_catalog'] = {}  # Gaia

            if idl_tel_method == 'spherical':
                fieldname_dict['reference_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_spherical_arcsec'
            else:
                fieldname_dict['reference_catalog']['position_1'] = 'v2_tangent_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_tangent_arcsec'
            if 'Name: J/A+A/563/A80/jwstcf' in reference_catalog.meta['comments']:
                reference_catalog_identifier = 'ID'  # HAWK-I
                fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error_mas'
                fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error_mas'
            else:
                reference_catalog_identifier = 'source_id'  # Gaia
                fieldname_dict['reference_catalog']['sigma_position_1'] = 'ra_error'
                fieldname_dict['reference_catalog']['sigma_position_2'] = 'dec_error'
            fieldname_dict['reference_catalog']['identifier'] = reference_catalog_identifier
            fieldname_dict['reference_catalog']['position_unit'] = u.arcsecond
            fieldname_dict['reference_catalog']['sigma_position_unit'] = u.milliarcsecond

            if idl_tel_method == 'spherical':
                fieldname_dict['star_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['star_catalog']['position_2'] = 'v3_spherical_arcsec'
            else:
                fieldname_dict['star_catalog']['position_1'] = 'v2_tangent_arcsec'
                fieldname_dict['star_catalog']['position_2'] = 'v3_tangent_arcsec'

            fieldname_dict['star_catalog']['sigma_position_1'] = 'sigma_x_mas'
            fieldname_dict['star_catalog']['sigma_position_2'] = 'sigma_y_mas'
            fieldname_dict['star_catalog']['identifier'] = 'star_id'
            fieldname_dict['star_catalog']['position_unit'] = u.arcsecond
            fieldname_dict['star_catalog']['sigma_position_unit'] = u.milliarcsecond

            obs.fieldname_dict = fieldname_dict

            observations.append(obs)

        pickle.dump((observations), open(parameters['pickle_file'], "wb"))
    else:
        observations = pickle.load(open(parameters['pickle_file'], "rb"))
        print('Loaded pickled file {}'.format(parameters['pickle_file']))

    return observations


def correct_dva(obs_collection, parameters):
    """Correct for effects of differential velocity aberration. This routine provides the necessary input to the DVA
    calculations (as attributes to the aperture object). DVA corrections are performed within the aperture methods when
    necessary.

    Parameters
    ----------
    obs_collection

    Returns
    -------

    """
    print('\nCORRECT FOR DIFFERENTIAL VELOCITY ABERRATION')

    dva_dir = parameters['dva_dir']
    dva_source_dir = parameters['dva_source_dir']
    verbose = parameters['verbose']

    for group_id in np.unique(obs_collection.T['group_id']):
        obs_indexes = np.where((obs_collection.T['group_id'] == group_id))[0]
        # obs_collection.T[obs_indexes].pprint()

        superfgs_observation_index = \
            np.where((obs_collection.T['group_id'] == group_id) & (
            obs_collection.T['INSTRUME'] == 'SUPERFGS'))[0]
        superfgs_obs = obs_collection.observations[superfgs_observation_index][0]

        camera_observation_index = \
            np.where((obs_collection.T['group_id'] == group_id) & (
            obs_collection.T['INSTRUME'] != 'SUPERFGS'))[0]

        fgs_exposure_midtimes = np.mean(np.vstack(
            (superfgs_obs.fpa_data['EXPSTART'].data, superfgs_obs.fpa_data['EXPEND'].data)), axis=0)

        # exclude HST FGS because it has already been corrected
        for i in camera_observation_index:

            camera_obs = obs_collection.observations[i]
            camera_obs_name_seed = '{}_{}_{}_{}_{}'.format(camera_obs.fpa_data.meta['TELESCOP'],
                                                           camera_obs.fpa_data.meta['INSTRUME'],
                                                           camera_obs.fpa_data.meta['APERTURE'],
                                                           camera_obs.fpa_data.meta['EPOCH'],
                                                           camera_obs.fpa_data.meta[
                                                               'DATAFILE'].split('.')[0]).replace(
                ':', '-')
            dva_filename = os.path.join(dva_dir, 'DVA_data_{}.txt'.format(camera_obs_name_seed))
            dva_file = open(dva_filename, 'w')
            # dva_file = sys.stdout

            camera_exposure_midtime = np.mean(
                [camera_obs.fpa_data.meta['EXPSTART'], camera_obs.fpa_data.meta['EXPEND']])
            matching_fgs_exposure_index = np.argmin(
                np.abs(camera_exposure_midtime - fgs_exposure_midtimes))
            if verbose:
                print('Camera-FGS Match found with delta_time = {:2.3f} min'.format(np.abs(
                    camera_exposure_midtime - fgs_exposure_midtimes[
                        matching_fgs_exposure_index]) * 24 * 60.))
                print('Writing parameter file for DVA code')
            for key in 'PRIMESI V2Ref V3Ref FGSOFFV2 FGSOFFV3 RA_V1 DEC_V1 PA_V3 POSTNSTX POSTNSTY POSTNSTZ VELOCSTX VELOCSTY VELOCSTZ EXPSTART'.split():
                if key in 'V2Ref V3Ref'.split():
                    value = getattr(superfgs_obs.aperture, key)
                elif key == 'PRIMESI':
                    value = np.int(superfgs_obs.fpa_data[key][matching_fgs_exposure_index][-1])
                elif key in 'FGSOFFV2 FGSOFFV3'.split():
                    value = superfgs_obs.fpa_data[key][matching_fgs_exposure_index]
                elif key == 'EXPSTART':
                    # Scale of EXPSTART seems to be UTC, see http://www.stsci.edu/ftp/documents/calibration/podps.dict
                    fgs_time = Time(camera_obs.fpa_data.meta[key], format='mjd', scale='utc')
                    value = fgs_time.yday.replace(':', ' ')
                else:
                    value = camera_obs.fpa_data.meta[key]

                print('{:<30} {}'.format(value, key), file=dva_file)
            dva_file.close()
            camera_obs.aperture._correct_dva = True
            camera_obs.aperture._dva_parameters = {'parameter_file': dva_filename,
                                                   'dva_source_dir': dva_source_dir}

            obs_collection.observations[i] = camera_obs
            if 0:

                v2, v3 = camera_obs.aperture.correct_for_dva(
                    np.array(camera_obs.star_catalog_matched['v2_spherical_arcsec']),
                    np.array(camera_obs.star_catalog_matched['v3_spherical_arcsec']))

                v2v3_data_file = os.path.join(dva_dir, 'v2v3_data_{}_measured.txt'.format(
                    camera_obs_name_seed))
                camera_obs.star_catalog_matched['v2_spherical_arcsec', 'v3_spherical_arcsec'].write(
                    v2v3_data_file,
                    format='ascii.fixed_width_no_header',
                    delimiter=' ',
                    bookend=False,
                    overwrite=True)

                v2v3_corrected_file = v2v3_data_file.replace('_measured', '_corrected')
                import subprocess

                system_command = '{} {} {} {}'.format(os.path.join(dva_source_dir, 'compute-DVA.e'),
                                                      dva_filename,
                                                      v2v3_data_file, v2v3_corrected_file)
                print('Running system command \n{}'.format(system_command))
                subprocess.call(system_command, shell=True)

                v2v3_corrected = Table.read(v2v3_corrected_file, format='ascii.no_header',
                                            names=('v2_original', 'v3_original', 'v2_corrected',
                                                   'v3_corrected'))

                1 / 0

                # interpolate ephmeris
                # fgs_time = Time(superfgs_obs.fpa_data[key][matching_fgs_exposure_index], format='mjd', scale='tdb')
                # start_time = Time(superfgs_obs.fpa_data[key][matching_fgs_exposure_index]-2001, format='mjd')
                # stop_time  = Time(superfgs_obs.fpa_data[key][matching_fgs_exposure_index]-1999, format='mjd')
                start_time = fgs_time - TimeDelta(1, format='jd')
                stop_time = fgs_time + TimeDelta(1, format='jd')

                # center = 'g@399'# Earth
                # center = '500@399'
                # target = '0' # SSB
                center = '500@0'  # SSB
                target = '399'  # Earth
                # target = '500@399'
                e = pystrometry.get_ephemeris(center=center, target=target, start_time=start_time,
                                              stop_time=stop_time,
                                              step_size='1h', verbose=True, out_dir=dva_dir,
                                              vector_table_output_type=2,
                                              output_units='KM-S', overwrite=True,
                                              reference_plane='FRAME')

                ip_values = []
                for colname in ['X', 'Y', 'Z', 'VX', 'VY', 'VZ']:
                    ip_fun = scipy.interpolate.interp1d(np.array(e['JDTDB']), np.array(e[colname]),
                                                        kind='linear', copy=True, bounds_error=True,
                                                        fill_value=np.nan)
                    # http://docs.astropy.org/en/stable/time/#id6
                    ip_val = ip_fun(fgs_time.tdb.jd)
                    ip_values.append(ip_val)

                e.add_row(np.hstack(([fgs_time.tdb.jd, 'N/A'], np.array(ip_values))))
                e[[-1]].pprint()
    return obs_collection


def hst_camera_fpa_data(data_dir, pattern, onepass_extension, standardized_data_dir, astrometry_uncertainty_mas):
    """
    Generate standardized focal plane alignment data based on HST camera data (WFC3, ACS) that were processed with
    Jay Anderson's hst1pass code to extract source pixel positions.


    ATTENTION: Pixel positions are extracted in individual chips (chips are numbered 1 and 2)
               hst1pass numbering is inverted compared to standard numbering scheme.

    chip 1 is the top chip
    chip 2 is the bottom chip

    :param data_dir:
    :param pattern:
    :param onepass_extension:
    :return:

    TODO: add FITS keywords needed for DVA correction from _spt header

    """

    invert_chip_numbers = True

    # file_list = glob.glob(os.path.join(data_dir, '**/*%s' % pattern))
    file_list = glob.glob(os.path.join(data_dir, '**/**/**/*{}'.format(pattern)))
    if len(file_list) == 0:
        raise RuntimeError('No HST camera data found')

    for f in file_list:
        # print(f)
        if '/logs/' in f:
            continue
        # get header
        primary_header = fits.getheader(f, ext=0)
        first_ext_header = fits.getheader(f, ext=1)

        # collect meta data
        telescope = primary_header['TELESCOP']
        instr = primary_header['INSTRUME'].strip()
        if instr == 'ACS':
            header_keyword_filter = 'FILTER1'
        elif instr == 'WFC3':
            header_keyword_filter = 'FILTER'
        filter = primary_header[header_keyword_filter].strip()
        aperture = primary_header['APERTURE'].strip()
        epoch_isot = '%sT%s' % (primary_header['DATE-OBS'], primary_header['TIME-OBS'])

        # data file
        # df = f.replace('.fits', '.%s' % onepass_extension)
        df = f.replace('.fits', '.%s' % onepass_extension).replace('mast_data', 'onepass_output')
        d = Table.read(df, format='ascii.basic', names=(list(onepass_extension)))

        print('Read {} stars from {}'.format(len(d), df))

        # extract keywords for DVA correction
        spt_file = f.replace('_flc.fits', '_spt.fits')


        d['dms_chip_number'] = np.zeros(len(d)).astype(np.int)
        onepass_chip_numbers = np.unique(d['k'].data)


        # construct DMS compliant chip numbers
        for chip_id in [1, 2]:
            chip_index = np.where(d['k'] == chip_id)[0]
            if chip_id == 1:
                if invert_chip_numbers:
                    d['dms_chip_number'][chip_index] = 2
                else:
                    d['dms_chip_number'][chip_index] = 1

            elif chip_id == 2:
                if invert_chip_numbers:
                    d['dms_chip_number'][chip_index] = 1
                else:
                    d['dms_chip_number'][chip_index] = 2

        dms_chip_numbers = np.unique(d['dms_chip_number'].data)
        chip_indices = []
        for chip_id in dms_chip_numbers:
            chip_index = np.where(d['dms_chip_number'] == chip_id)[0]
            chip_indices.append(chip_index)

        # flt images are in SIAS. hst1pass x,y coordinates are therefore also in SIAS
        d['x_SCI'] = d['x']
        d['y_SCI'] = d['y']

        # index of chip 1
        if invert_chip_numbers:
            chip_correct_index = dms_chip_numbers.tolist().index(1)
        else:
            chip_correct_index = dms_chip_numbers.tolist().index(2)

        # correct y coordinates of chip1
        d['y_SCI'][chip_indices[chip_correct_index]] = d['y'][chip_indices[chip_correct_index]] - 2048.

        d['RA_deg'] = d['r']
        d['Dec_deg'] = d['d']

        if type(astrometry_uncertainty_mas) is dict:
            exp_time = primary_header['EXPTIME']
            if exp_time < astrometry_uncertainty_mas['shallow']['exptime_threshold_s']:
                uncertainty_mas = astrometry_uncertainty_mas['shallow']['uncertainty_mas']
            elif exp_time > astrometry_uncertainty_mas['deep']['exptime_threshold_s']:
                uncertainty_mas = astrometry_uncertainty_mas['deep']['uncertainty_mas']

            d['sigma_x_mas'] = np.ones(len(d)) * uncertainty_mas
            d['sigma_y_mas'] = np.ones(len(d)) * uncertainty_mas

        else:
            d['sigma_x_mas'] = np.ones(len(d)) * astrometry_uncertainty_mas
            d['sigma_y_mas'] = np.ones(len(d)) * astrometry_uncertainty_mas

        # clean Table
        d2 = Table()
        for col in d.colnames:
            d2[col] = d[col]

        header_keys = ['TELESCOP', 'INSTRUME', 'PA_V3', 'APERTURE', 'DATE-OBS', 'TIME-OBS', 'EXPSTART', 'EXPEND', 'PROPOSID', 'EXPTIME']
        # PA_V3   =           269.998413 / position angle of V3-axis of HST (deg)
        for key in header_keys:
            d2.meta[key] = primary_header[key]

        spt_header = fits.getheader(spt_file)
        for key in 'RA_V1 DEC_V1 POSTNSTX POSTNSTY POSTNSTZ VELOCSTX VELOCSTY VELOCSTZ APER_REF APERTYPE DGESTAR SGESTAR'.split():
            d2.meta[key] = spt_header[key]
            # RA_V1   =   9.230892548311E+01 / right ascension of v1 axis of st (deg)
            # DEC_V1  =   2.434359995048E+01 / declination of v1 axis of st (deg)
            # APER_REF= 'JWFCFIX   '         / aperture used for reference position
            # APERTYPE= 'SICS              ' / aperture type (SICS, SIAS, SIDS)

        d2.meta['FILTER'] = primary_header[header_keyword_filter].strip()
        d2.meta['DATAFILE'] = os.path.basename(f)
        d2.meta['DATAPATH'] = os.path.dirname(f)
        d2.meta['EPOCH'] = epoch_isot
        header2_keys = ['RA_APER', 'DEC_APER', 'PA_APER']
        # first_ext_header
        # PA_APER =              87.5009 / Position Angle of reference aperture center (de
        # RA_APER =   9.223601092740E+01 / RA of aperture reference position
        # DEC_APER=   2.441575887367E+01 / Declination of aperture reference position

        # primary header
        # RA_TARG =   9.223601092740E+01 / right ascension of the target (deg) (J2000)
        for key in header2_keys:
            d2.meta[key] = first_ext_header[key]

        # == APER_REF !
        if d2.meta['APERTURE'] == 'WFC-FIX':
            d2.meta['SIAFAPER'] = 'JWFCFIX'
        elif d2.meta['APERTURE'] == 'UVIS-CENTER':
            d2.meta['SIAFAPER'] = 'IUVISCTR'

        d2.meta['PROGRAM_VISIT'] = '{}_{}'.format(d2.meta['PROPOSID'], d2.meta['DATAFILE'][4:6])

        for j, chip_id in enumerate(dms_chip_numbers):
            out_file = os.path.join(standardized_data_dir, 'FPA_data_%s_%s_%s_chip%d_%s_%s_%s.fits' % (
            telescope, instr, aperture, chip_id, filter, epoch_isot, os.path.basename(f).split('.')[0])).replace(':',
                                                                                                                 '-')
            print('Writing {} ({} stars)'.format(out_file, len(chip_indices[j])))
            d2.meta['CHIP'] = chip_id
            d2[chip_indices[j]].write(out_file, overwrite=True)


def jwst_camera_fpa_data(data_dir, pattern, standardized_data_dir, parameters,
                         overwrite_source_extraction=False):
    """Generate standardized focal plane alignment data based on JWST camera data.
    """

    # invert_chip_numbers = True

    # file_list = glob.glob(os.path.join(data_dir, '**/*%s' % pattern))
    # file_list = glob.glob(os.path.join(data_dir, '**/**/**/*{}'.format(pattern)))
    file_list = glob.glob(os.path.join(data_dir, '*{}'.format(pattern)))

    if len(file_list) == 0:
        raise RuntimeError('No data found')

    for f in file_list:
        # if 'jw01088002001_01201_00004_g2_cal' not in f:
        # if 'jw01087001001_01101_00031_nis_cal' not in f:
        #     continue

        pl.close('all')
        print('processing {}'.format(f))

        im = datamodels.open(f)
        if hasattr(im, 'data') is False:
            im.data = fits.getdata(f)
            im.dq = np.zeros(im.data.shape)

        header_info = OrderedDict()

        for attribute in 'telescope'.split():
            header_info[attribute] = getattr(im.meta, attribute)

        # observations
        for attribute in 'date time visit_number visit_id visit_group activity_id program_number'.split():
            header_info['observation_{}'.format(attribute)] = getattr(im.meta.observation, attribute)

        header_info['epoch_isot'] = '{}T{}'.format(header_info['observation_date'], header_info['observation_time'])

        #  instrument
        for attribute in 'name filter pupil detector'.split():
            header_info['instrument_{}'.format(attribute)] = getattr(im.meta.instrument, attribute)

        # subarray
        for attribute in 'name'.split():
            header_info['subarray_{}'.format(attribute)] = getattr(im.meta.subarray, attribute)

        # aperture
        for attribute in 'name position_angle pps_name'.split():
            try:
                value = getattr(im.meta.aperture, attribute)
            except AttributeError:
                value = None

            header_info['aperture_{}'.format(attribute)] = value


        # temporary solution, this should come from pupulated aperture attributes
        if header_info['subarray_name'] == 'FULL':
            master_apertures = pysiaf.read.read_siaf_detector_layout()
            if header_info['instrument_name'].lower() in ['niriss', 'miri']:
                header_info['SIAFAPER'] = master_apertures['AperName'][np.where(master_apertures['InstrName']==header_info['instrument_name'])[0][0]]
            elif header_info['instrument_name'].lower() in ['fgs']:
                header_info['SIAFAPER'] = 'FGS{}_FULL'.format(header_info['instrument_detector'][-1])

        # target
        for attribute in 'ra dec catalog_name proposer_name'.split():
            header_info['target_{}'.format(attribute)] = getattr(im.meta.target, attribute)

        # pointing
        for attribute in 'ra_v1 dec_v1 pa_v3'.split():
            try:
                value = getattr(im.meta.pointing, attribute)
            except AttributeError:
                value = None
            header_info['pointing_{}'.format(attribute)] = value

        # add HST style keywords
        header_info['PROGRAM_VISIT'] = '{}_{}'.format(header_info['observation_program_number'], header_info['observation_visit_id'])
        header_info['PROPOSID'] = header_info['observation_program_number']
        header_info['DATE-OBS'] = header_info['observation_date']
        header_info['TELESCOP'] = header_info['telescope']
        header_info['INSTRUME'] = header_info['instrument_name']
        try:
            header_info['APERTURE'] = header_info['SIAFAPER']
        except KeyError:
            header_info['APERTURE'] = None
        header_info['CHIP'] = 0

        extracted_sources_dir = os.path.join(standardized_data_dir, 'extraction')
        if os.path.isdir(extracted_sources_dir) is False:
            os.makedirs(extracted_sources_dir)
        extracted_sources_file = os.path.join(extracted_sources_dir,
                                              '{}_extracted_sources.fits'.format(os.path.basename(f)))

        mask_extreme_slope_values = False
        parameters['maximum_slope_value'] = 1000.
        # parameters['use_epsf'] = True
        # parameters['show_extracted_sources'] = True

        # parameters['use_DAOStarFinder_for_epsf'] = False
        # parameters['use_DAOStarFinder_for_epsf'] = True

        if (not os.path.isfile(extracted_sources_file)) or (overwrite_source_extraction):
            show_figures = False
            data = copy.deepcopy(im.data)
            dq = copy.deepcopy(im.dq)

            if mask_extreme_slope_values:
                # clean up extreme slope values
                bad_index = np.where(np.abs(data) > parameters['maximum_slope_value'])
                data[bad_index] = 0.
                dq[bad_index] = -1
            # pl.figure()
            # pl.hist(np.abs(data.flatten()), 100)
            # pl.show()
            # 1/0
            # data[np.where(im.dq!=0)] = 0.
            # pl.figure()
            # pl.hist(data.flatten(), 100)
            # pl.show()
            # 1/0

            mean, median, std = sigma_clipped_stats(data, sigma=3.0)


            import corner
            if parameters['use_epsf'] is False:
                daofind = DAOStarFinder(fwhm=2.0, threshold=parameters['dao_detection_threshold'] * std)
                # daofind = DAOStarFinder(fwhm=2.0, threshold=10. * std)
                dao_extracted_sources = daofind(data - median)
            # extracted_sources.write(extracted_sources_file, overwrite=True)
            # extracted_sources = extracted_sources[extracted_sources['peak']<500]
                if dao_extracted_sources is None:
                    dao_extracted_sources = Table()
                else:
                    print('Initial source extraction: {} sources'.format(len(dao_extracted_sources)))
                    if 0:
                        corner_plot_file = os.path.join(extracted_sources_dir,
                                                    '{}_corner_extraction.pdf'.format(os.path.basename(f).split('.')[0]))
                        selected_columns = [col for col in dao_extracted_sources.colnames if col not in 'npix sky'.split()]
                        samples = np.array([dao_extracted_sources[col] for col in selected_columns])
                        # title_string = '{}: {} extracted sources'.format(os.path.basename(f), len(extracted_sources))
                        # fig = pl.figure()
                        fig = corner.corner(samples.T, labels=selected_columns)
                        # pl.text(0.5, 0.95, title_string, horizontalalignment='center', verticalalignment = 'center', transform = pl.gca().transAxes)
                        fig.savefig(corner_plot_file)
                        if show_figures:
                            pl.show()
                    dao_extracted_sources = dao_extracted_sources[(np.abs(dao_extracted_sources['roundness1'])<parameters['roundness_threshold'])
                                                          & (dao_extracted_sources['sharpness']>parameters['sharpness_threshold'])]
                                                          # & (np.abs(extracted_sources['roundness2'])<parameters['roundness_threshold'])
                    dao_extracted_sources.write(extracted_sources_file, overwrite=True)
                    print('Sharpness/roundness cut: {} sources'.format(len(dao_extracted_sources)))
                extracted_sources = dao_extracted_sources




            else:
                detection_fwhm =2.0
                detection_threshold = parameters['dao_detection_threshold'] * std

                epsf_psf_size_pix = parameters['epsf_psf_size_pix']

                nearest_neighbour_distance_threshold_pix = epsf_psf_size_pix

                # EPSF builder
                if parameters['use_DAOStarFinder_for_epsf']: # use DAOStarFinder for EPSF star list

                    # daofind = DAOStarFinder(fwhm=2.0, threshold=20. * std)
                    # daofind = DAOStarFinder(fwhm=2.0, threshold=5. * std)
                    daofind = DAOStarFinder(fwhm=detection_fwhm, threshold=detection_threshold)
                    # daofind = DAOStarFinder(fwhm=2.0, threshold=3. * std)
                    dao_extracted_sources = daofind(data - median)

                    print('Initial source extraction: {} sources'.format(len(dao_extracted_sources)))
                    if 1:
                        corner_plot_file = os.path.join(extracted_sources_dir,
                                                        '{}_corner_extraction.pdf'.format(
                                                            os.path.basename(f).split('.')[0]))
                        selected_columns = [col for col in dao_extracted_sources.colnames if
                                            col not in 'npix sky'.split()]
                        samples = np.array([dao_extracted_sources[col] for col in selected_columns])
                        fig = corner.corner(samples.T, labels=selected_columns)
                        fig.savefig(corner_plot_file)
                        pl.show()
                        # 1 / 0

                    dao_extracted_sources = select_isolated_sources(dao_extracted_sources, nearest_neighbour_distance_threshold_pix)
                    dao_extracted_sources = dao_extracted_sources[(np.abs(dao_extracted_sources['roundness1'])<parameters['roundness_threshold'])
                                                          & (dao_extracted_sources['sharpness']>parameters['sharpness_threshold'])]
                    print('Sharpness/roundness + isolation cut: {} sources'.format(len(dao_extracted_sources)))

                    if 1:
                        dao_extracted_sources.remove_rows(
                            np.where(dao_extracted_sources['flux'] <= 0)[0])
                        # flux_threshold_percentile = 50
                        # flux_threshold = np.percentile(dao_extracted_sources['flux'], flux_threshold_percentile)
                        flux_threshold_lower = np.percentile(dao_extracted_sources['flux'], parameters['flux_threshold_percentile_lower'])
                        flux_threshold_upper = np.percentile(dao_extracted_sources['flux'], parameters['flux_threshold_percentile_upper'])
                        # dao_extracted_sources.remove_rows(np.where(dao_extracted_sources['flux'] < flux_threshold)[0])
                        dao_extracted_sources.remove_rows(np.where(dao_extracted_sources['flux'] < flux_threshold_lower)[0])
                        dao_extracted_sources.remove_rows(np.where(dao_extracted_sources['flux'] > flux_threshold_upper)[0])
                        # print('Only {} sources have positve flux > {:2.3f}'.format(len(dao_extracted_sources), flux_threshold))
                        print('Only {} sources have positve {:2.3f} > flux > {:2.3f}'.format(len(dao_extracted_sources), flux_threshold_upper, flux_threshold_lower))
                    # 1/0
                else:
                    from photutils import find_peaks
                    from photutils.centroids import centroid_2dg

                    # extracted_sources = find_peaks(data, threshold=75, box_size=25, centroid_func=centroid_2dg, mask=im.dq!=0 )
                    extracted_sources = find_peaks(data - median, threshold=10. * std, box_size=10,
                                                   centroid_func=centroid_2dg, mask=im.dq!=0 )
                    extracted_sources.rename_column('x_centroid', 'xcentroid')
                    extracted_sources.rename_column('y_centroid', 'ycentroid')
                    print('Initial source extraction using find_peaks: {} sources'.format(len(extracted_sources)))




                # for j in range(len(extracted_sources)):

                # epsf_psf_size_pix = 25
                # epsf_psf_size_pix = 15

                # see https://photutils.readthedocs.io/en/stable/epsf.html
                size = epsf_psf_size_pix + 10
                hsize = (size - 1) / 2
                x = dao_extracted_sources['xcentroid']
                y = dao_extracted_sources['ycentroid']
                mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) & (y > hsize) & (y < (data.shape[0] - 1 - hsize)))

                stars_tbl = Table()
                stars_tbl['x'] = x[mask]
                stars_tbl['y'] = y[mask]
                mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
                print('{}: mean {:2.3f} median {:2.3f} std {:2.3f}'.format(f, mean_val, median_val, std_val))
                # 1/0
                # data -= median_val

                # stars_tbl = stars_tbl[0:5]
                # stars_tbl = stars_tbl[[0]]
                print('Using {} stars to build epsf'.format(len(stars_tbl)))

                from astropy.nddata import NDData
                nddata = NDData(data=data-median_val)

                if 0:
                    from astropy.nddata import StdDevUncertainty, NDUncertainty
                    std_uncertainty = StdDevUncertainty(data)

                    # weight_uncertainty = copy.deepcopy(std_uncertainty)
                    weight_uncertainty = NDUncertainty(data=std_uncertainty.array, uncertainty_type='weights')
                    weight_uncertainty.array = 1./(std_uncertainty.array**2)
                    weight_uncertainty.array[np.where(dq.data != 0)] = 0
                    weight_uncertainty.uncertainty_type = 'weights'

                    # np.where(dq.data != 0)
                    nddata = NDData(data=data-median_val, uncertainty=weight_uncertainty)#, uncertainty_type='weights')



                from photutils.psf import extract_stars
                stars = extract_stars(nddata, stars_tbl, size=epsf_psf_size_pix)


                # use_weights_for_epsf = True
                if parameters['use_weights_for_epsf']:
                    dqs = extract_stars(NDData(data=dq), stars_tbl, size=epsf_psf_size_pix)
                    # print([np.any(dqs[i].data!=0) for i in range(len(dqs))] )
                    # # dqs[0].data[]
                    # print([len(np.where(dqs[i].data!=0)[0]) for i in range(len(dqs))]   )
                    for j,star in enumerate(stars):
                        # star.weights = 1./np.
                        mask_index = np.where(dqs[j].data!=0)
                        star.weights[mask_index] = 0


                        # 1/0


                import matplotlib.pyplot as plt
                from astropy.visualization import simple_norm
                nrows = 10
                ncols = nrows
                # nrows = 10
                # ncols = 10
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
                ax = ax.ravel()
                for i in range(nrows * ncols):
                    if i <= len(stars)-1:
                        norm = simple_norm(stars[i], 'log', percent=99.)
                        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
                # pl.show()
                pl.title('{} sample stars for epsf'.format(header_info['APERTURE']))
                psf_plot_file = os.path.join(extracted_sources_dir,
                                                '{}_sample_psfs.pdf'.format(
                                                    os.path.basename(f).split('.')[0]))
                pl.savefig(psf_plot_file)

                from photutils import EPSFBuilder
                from photutils.centroids import centroid_com
                epsf_builder = EPSFBuilder(oversampling=4, maxiters=3, recentering_maxiters=5, recentering_func=centroid_com)
                # epsf_builder = EPSFBuilder(oversampling=4, progress_bar=True)
                print('Building epsf ...')
                # epsf, fitted_stars = epsf_builder(stars)
                epsf, fitted_stars = epsf_builder(stars)

                norm = simple_norm(epsf.data, 'log', percent=99.)
                pl.figure()
                pl.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
                pl.colorbar()
                # pl.show()
                pl.title('{} epsf using {} stars'.format(header_info['APERTURE'], len(stars_tbl)))
                epsf_plot_file = os.path.join(extracted_sources_dir,
                                                '{}_epsf.pdf'.format(
                                                    os.path.basename(f).split('.')[0]))
                pl.savefig(epsf_plot_file)

                # 1/0
                from photutils.psf import IntegratedGaussianPRF, DAOGroup
                from photutils.background import MMMBackground, MADStdBackgroundRMS
                from astropy.modeling.fitting import LevMarLSQFitter
                from astropy.stats import gaussian_sigma_to_fwhm

                sigma_psf = 2.0
                image = data
                # bkgrms = MADStdBackgroundRMS()
                # std = bkgrms(image)
                # iraffind = IRAFStarFinder(threshold=20 * std,
                #                           fwhm=sigma_psf * gaussian_sigma_to_fwhm, minsep_fwhm=1.0,
                #                           roundhi=1.0, roundlo=-1.0, sharplo=0.5, sharphi=2.0,
                #                           )
                                          # peakmax=parameters['maximum_slope_value'])

                daofind2 = DAOStarFinder(threshold=detection_threshold, fwhm=detection_fwhm,
                                         roundhi=parameters['roundness_threshold'], sharplo=parameters['sharpness_threshold'])


                daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
                mmm_bkg = MMMBackground()
                # fitter = LevMarLSQFitter()
                # psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
                psf_model = epsf.copy()
                from photutils.psf import IterativelySubtractedPSFPhotometry
                photometry = IterativelySubtractedPSFPhotometry(finder=daofind2, #iraffind,
                                                                group_maker=daogroup,
                                                                bkg_estimator=mmm_bkg,
                                                                psf_model=psf_model,
                                                                fitter=LevMarLSQFitter(), niters=1,
                                                                fitshape=(11, 11), aperture_radius=5,)
                                                                # extra_output_cols=['roundness1', 'sharpness', 'roundness2'])

                print('Performing source extraction and photometry ...')
                # data1 = image[0:200, 0:200]
                epsf_extracted_sources = photometry(image=image)
                # result_tab = photometry(image=data1)
                # print(result_tab)
                print('Final source extraction with epsf: {} sources'.format(len(epsf_extracted_sources)))
                epsf_extracted_sources['xcentroid'] = epsf_extracted_sources['x_fit']
                epsf_extracted_sources['ycentroid'] = epsf_extracted_sources['y_fit']

                extracted_sources = epsf_extracted_sources
                if 0:
                    corner_plot_file = os.path.join(extracted_sources_dir,
                                                '{}_corner_epsf_extraction.pdf'.format(os.path.basename(f).split('.')[0]))
                    selected_columns = [col for col in epsf_extracted_sources.colnames if col not in 'npix sky iter_detected'.split()]
                    samples = np.array([epsf_extracted_sources[col] for col in selected_columns])
                    title_string = '{}: {} epsf extracted sources'.format(os.path.basename(f), len(extracted_sources))
                    # fig = pl.figure()
                    fig = corner.corner(samples.T, labels=selected_columns)
                    # pl.text(0.5, 0.95, title_string, horizontalalignment='center', verticalalignment = 'center', transform = pl.gca().transAxes)
                    fig.savefig(corner_plot_file)
                # 1/0

                if 0:
                    print('Making residual image ...')
                    residual_image = photometry.get_residual_image()
                    pl.figure()
                    plt.subplot(1, 2, 1)
                    plt.imshow(image, cmap='viridis', aspect=1, interpolation='nearest')
                    plt.title('Simulated data')
                    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
                    plt.subplot(1, 2, 2)
                    plt.imshow(residual_image, cmap='viridis', aspect=1, interpolation='nearest',
                               origin='lower')
                    plt.title('Residual Image')
                    plt.colorbar(orientation='horizontal', fraction=0.046, pad=0.04)
                    plt.show()
                    1/0

            extracted_sources.write(extracted_sources_file, overwrite=True)

            if parameters['show_extracted_sources']:
                # data[np.where(np.abs(data)>65000)] = np.nan
                data[np.where(np.abs(data)>100)] = 0.1
                # print(extracted_sources)

                positions = (extracted_sources['xcentroid'], extracted_sources['ycentroid'])
                apertures = CircularAperture(positions, r=4.)
                norm = ImageNormalize(stretch=SqrtStretch())
                # norm = ImageNormalize(stretch=LogStretch())
                extracted_plot_file = os.path.join(extracted_sources_dir,
                                                '{}_extracted_sources.pdf'.format(
                                                    os.path.basename(f).split('.')[0]))

                fig = pl.figure()
                pl.imshow(data, cmap='Greys', origin='lower', norm=norm)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)

                epsf_positions = (epsf_extracted_sources['xcentroid'], epsf_extracted_sources['ycentroid'])
                epsf_apertures = RectangularAperture(epsf_positions, w=6, h=6)
                epsf_apertures.plot(color='green', lw=1.5, alpha=0.5)

                title_string = '{}: {} selected sources'.format(os.path.basename(f),
                                                                 len(extracted_sources))
                pl.title(title_string)
                if show_figures:
                    pl.show()
                # pl.show()
                fig.savefig(extracted_plot_file)
                # 1/0
        else:
            extracted_sources = Table.read(extracted_sources_file)



        print('Extracted {} sources from {}'.format(len(extracted_sources), f))
        impose_positive_flux = True
        if impose_positive_flux and parameters['use_epsf']:
            extracted_sources.remove_rows(np.where(extracted_sources['flux_fit']<0)[0])
            print('Only {} sources have positve flux'.format(len(extracted_sources)))

            if 0:
                pl.figure()
                pl.hist(extracted_sources['flux_fit'], 100)
                pl.show()

            if 0:
                flux_threshold_percentile = 50
                flux_threshold = np.percentile(extracted_sources['flux_fit'], flux_threshold_percentile)
                # extracted_sources.remove_rows(np.where(extracted_sources['flux_fit'] > flux_threshold)[0])
                extracted_sources.remove_rows(np.where(extracted_sources['flux_fit'] < flux_threshold)[0])
                # print('Only {} sources have positve flux < {:2.3f}'.format(len(extracted_sources), flux_threshold))
                print('Only {} sources have positve flux > {:2.3f}'.format(len(extracted_sources), flux_threshold))

            # 1/0


        # data file
        # df = f.replace('.fits', '.%s' % onepass_extension)
        # df = f.replace('.fits', '.%s' % onepass_extension).replace('mast_data', 'onepass_output')
        # d = Table.read(df, format='ascii.basic', names=(list(onepass_extension)))


        # extract keywords for DVA correction
        # spt_file = f.replace('_flc.fits', '_spt.fits')


        # d['dms_chip_number'] = np.zeros(len(d)).astype(np.int)
        # onepass_chip_numbers = np.unique(d['k'].data)
        astrometry_uncertainty_mas = 5

        if len(extracted_sources) > 0:
            # cal images are in DMS coordinates. These correspond to the SIAF Science (SCI) frame
            extracted_sources['x_SCI'], extracted_sources['y_SCI'] = extracted_sources['xcentroid'], extracted_sources['ycentroid']

            # d['RA_deg'] = d['r']
            # d['Dec_deg'] = d['d']

            # if type(astrometry_uncertainty_mas) is dict:
            #     exp_time = primary_header['EXPTIME']
            #     if exp_time < astrometry_uncertainty_mas['shallow']['exptime_threshold_s']:
            #         uncertainty_mas = astrometry_uncertainty_mas['shallow']['uncertainty_mas']
            #     elif exp_time > astrometry_uncertainty_mas['deep']['exptime_threshold_s']:
            #         uncertainty_mas = astrometry_uncertainty_mas['deep']['uncertainty_mas']
            #
            #     d['sigma_x_mas'] = np.ones(len(d)) * uncertainty_mas
            #     d['sigma_y_mas'] = np.ones(len(d)) * uncertainty_mas
            #
            # else:
            extracted_sources['sigma_x_mas'] = np.ones(len(extracted_sources)) * astrometry_uncertainty_mas
            extracted_sources['sigma_y_mas'] = np.ones(len(extracted_sources)) * astrometry_uncertainty_mas

        # transfer info to astropy table header
        for key, value in header_info.items():
            extracted_sources.meta[key] = value


        if 0:
            # # clean Table
            # d2 = Table()
            # for col in d.colnames:
            #     d2[col] = d[col]

            header_keys = ['TELESCOP', 'INSTRUME', 'PA_V3', 'APERTURE', 'DATE-OBS', 'TIME-OBS', 'EXPSTART', 'EXPEND', 'PROPOSID', 'EXPTIME']
            # PA_V3   =           269.998413 / position angle of V3-axis of HST (deg)
            for key in header_keys:
                d2.meta[key] = primary_header[key]

            spt_header = fits.getheader(spt_file)
            for key in 'RA_V1 DEC_V1 POSTNSTX POSTNSTY POSTNSTZ VELOCSTX VELOCSTY VELOCSTZ APER_REF APERTYPE DGESTAR SGESTAR'.split():
                d2.meta[key] = spt_header[key]
                # RA_V1   =   9.230892548311E+01 / right ascension of v1 axis of st (deg)
                # DEC_V1  =   2.434359995048E+01 / declination of v1 axis of st (deg)
                # APER_REF= 'JWFCFIX   '         / aperture used for reference position
                # APERTYPE= 'SICS              ' / aperture type (SICS, SIAS, SIDS)

            d2.meta['FILTER'] = primary_header[header_keyword_filter].strip()
        extracted_sources.meta['DATAFILE'] = os.path.basename(f)
        extracted_sources.meta['DATAPATH'] = os.path.dirname(f)
        extracted_sources.meta['EPOCH'] = header_info['epoch_isot']

        if 0:
            header2_keys = ['RA_APER', 'DEC_APER', 'PA_APER']
            # first_ext_header
            # PA_APER =              87.5009 / Position Angle of reference aperture center (de
            # RA_APER =   9.223601092740E+01 / RA of aperture reference position
            # DEC_APER=   2.441575887367E+01 / Declination of aperture reference position

            # primary header
            # RA_TARG =   9.223601092740E+01 / right ascension of the target (deg) (J2000)
            for key in header2_keys:
                d2.meta[key] = first_ext_header[key]

            # == APER_REF !
            if d2.meta['APERTURE'] == 'WFC-FIX':
                d2.meta['SIAFAPER'] = 'JWFCFIX'
            elif d2.meta['APERTURE'] == 'UVIS-CENTER':
                d2.meta['SIAFAPER'] = 'IUVISCTR'

            d2.meta['PROGRAM_VISIT'] = '{}_{}'.format(d2.meta['PROPOSID'], d2.meta['DATAFILE'][4:6])

        out_file = os.path.join(standardized_data_dir, 'FPA_data_{}_{}_{}_{}_{}_{}_{}.fits'.format(extracted_sources.meta['telescope'],
                                                                                             extracted_sources.meta['instrument_name'],
                                                                                             extracted_sources.meta['subarray_name'],
                                                                                             extracted_sources.meta['instrument_filter'],
                                                                                             extracted_sources.meta['instrument_pupil'],
                                                                                             extracted_sources.meta['EPOCH'].replace(':','-').replace('.','-'),
                                                                                             extracted_sources.meta['DATAFILE'].split('.')[0]).replace('/',''))
        print('Writing {}'.format(out_file))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning, append=True)
            extracted_sources.write(out_file, overwrite=True)

    return im

def hst_guider_fpa_data(reduced_data_dir, mast_data_dir, pattern, standardized_data_dir, verbose=True):
    """
    Generate standardized focal plane alignment data based on HST guider data (FGS1, FGS2, FGS3)

    :param data_dir:
    :param pattern:
    :param standardized_data_dir:
    :param astrometry_uncertainty_mas:
    :return:
    """
    # file_list = glob.glob(os.path.join(data_dir, '**/*%s' % pattern))
    file_list = glob.glob(os.path.join(reduced_data_dir, '*{}*'.format(pattern)))
    for f in file_list:

        if verbose:
            print('Reading Guider data file: {}'.format(f))

        # d = Table.read(f, format='ascii.basic', delimiter= '\t', guess=False)
        # d = Table.read(f, format='ascii.basic', delimiter= ',', guess=False)
        d = Table.read(f, format='ascii.basic', delimiter= ',', guess=False, data_start=2)
        # d = Table.read(f, format='ascii.fixed_width', delimiter= ' ', guess=False)
        # d = Table.read(f, format='ascii.basic', guess=False)

        if verbose:
            d.pprint()
        # d = Table.read(f, format='ascii.basic')
        # 1/0
        # loop over individual FGS frames
        for j, obs_id in enumerate(d['OBS_ID'].data):

            # retrieve basic exposure data from MAST
            mast_data = mast.Observations.query_criteria(obs_id=obs_id.upper(), obstype='cal')
            # if 0:
            #
            #     # mast_data = mast.Observations.query_criteria(obs_id=d['obsID'].data)
            #
            #     # d = hstack((d, mast_data))
            #     mast_dir = os.path.join(data_dir, 'FGS/mast')
            #     if overwrite_mast_files:
            #         dataProductsByObservation = mast.Observations.get_product_list(mast_data)
            #         mast.Observations.download_products(dataProductsByObservation, mrp_only=False, download_dir=mast_dir)
            #
            #     download_dir = os.path.join(mast_dir, 'mastDownload/HST', obs_id)

            download_dir = mast_data_dir

            # a1f_file = glob.glob( os.path.join(download_dir, '*%s'%'_a1f.fits'))[0]
            # a1f_file = glob.glob( os.path.join(download_dir, '**/**/**/*%s'%'_a1f.fits'))
            a1f_file = glob.glob( os.path.join(download_dir, '**/**/FGS/{}{}.fits'.format(obs_id.lower(), '_a1f')))
            if len(a1f_file) != 1:
                raise RuntimeError('Identified NO or more than one match for {}'.format(obs_id))
            else:
                a1f_file = a1f_file[0]
            if verbose:
                print('Identifed _a1f file: {}'.format(a1f_file))
            a1f_header = fits.getheader(a1f_file)

            # primary instrument
            instr = a1f_header['PRIMESI'].strip()
            fgs_number = np.int(instr.split('FGS')[1])

            af_header = fits.getheader(a1f_file.replace('_a1f', '_a%df'%fgs_number))
            # h = af_header

            # read DMF header to get RA/Dec of V1
            dmf_header = fits.getheader(a1f_file.replace('_a1f', '_dmf'))

            # clean Table
            d2 = Table()
            for col in d.colnames:
                d2[col] = d[col][[j]]
            for col in mast_data.colnames:
                d2[col] = mast_data[col]

            # 1/0
            d2['RA_deg'] = d2['RA']
            d2['Dec_deg'] = d2['DEC']


            # nelan email dated 20 October 2017:
            # The (x,y) values are corrected for geometric distortion, differential velocity aberration, spacecraft jitter, and spacecraft drift.
            # question is whether these are cartesian or polar coordinates
            d2['x_idl_arcsec'] = d2['X']
            d2['y_idl_arcsec'] = d2['Y']

            d2['sigma_x_mas'] = np.array(d2['X_sd']) * 1000.
            d2['sigma_y_mas'] = np.array(d2['X_sd']) * 1000.

            header_keys = ['TELESCOP', 'INSTRUME', 'DATE-OBS', 'TIME-OBS'] #'PA_V3',
            for key in header_keys:
                d2.meta[key] = af_header[key]

            dmf_keys = 'PA_V3 RA_V1 DEC_V1 DGESTAR SGESTAR'.split()
            for key in dmf_keys:
                d2.meta[key] = dmf_header[key]
            # d2.meta['PA_V3'] = dmf_header['PA_V3'] # / position angle of V3-axis of HST (deg) this is at V2,V3=0,0
            # # see /Users/jsahlmann/jwst/code/github/spacetelescope/mirage/jwst/jwst/lib/set_telescope_pointing.py
            # # V3APERCE=           326.492096 / V3 offset of target from aper fiducial (arcsec)
            # d2.meta['RA_V1'] = dmf_header['RA_V1'] # right ascension of v1 axis of st (deg)
            # d2.meta['DEC_V1'] = dmf_header['DEC_V1'] # declination of v1 axis of st (deg)
            d2.meta['PA_APER'] = d['PA'][j]

            telescope = af_header['TELESCOP']
            header_keyword_filter = 'filters'
            d2.meta['FILTER'] = d2[header_keyword_filter][0]
            d2.meta['DATAFILE'] = os.path.basename(f)

            epoch_isot = '%sT%s' % (af_header['DATE-OBS'], af_header['TIME-OBS'])
            d2.meta['EPOCH'] = epoch_isot
            # header2_keys = ['RA_APER', 'DEC_APER', 'PA_APER']
            # for key in header2_keys:
            #     d2.meta[key] = h[key]


            d2.meta['SIAFAPER'] = instr
            aperture = d2.meta['SIAFAPER']
            d2.meta['APERTURE'] = aperture
            filter = d2.meta['FILTER']
            chip_id = 0

            out_file = os.path.join(standardized_data_dir, 'FPA_data_%s_%s_%s_chip%d_%s_%s_%s.fits' % (
                    telescope, instr, aperture, chip_id, filter, epoch_isot,
                    os.path.basename(f).split('.')[0])).replace(':','-')
            print('Writing %s' % out_file)
            d2.write(out_file, overwrite=True)

            for key in 'PRIMESI PROPOSID'.split():
                d2[key] = a1f_header[key]
            for key in 'FGSOFFV2 FGSOFFV3 FGS_PAV3 FGSREFV2 FGSREFV3 PVEL_AB EXPTIME'.split():
                d2[key] = af_header[key]

            # VELABBRA 4.052049 / aberration in position of the target
            # V2APERCE=            10.000000 / V2 offset of target from aper fiducial (arcsec)
            # POSTNSTX=   4.755696683602E+03 / position of space telescope x axis (km)
            # EXPSTART=       57855.30789390 / exposure start time (Modified Julian Date)
            for key in 'VELOCSTX VELOCSTY VELOCSTZ VELABBRA V2APERCE V3APERCE POSTNSTX POSTNSTY POSTNSTZ EXPSTART EXPEND PA_V3 RA_V1 DEC_V1'.split():
                # print('dmf_header: {} = {}'.format(key, dmf_header[key]))
                # d2.meta[key] = dmf_header[key]
                d2[key] = dmf_header[key]



            # if dmf_header['EXPSTART'] == 57855.4404865: # Ed's example file 	Xpos does not seem to match POSTNSTX
            #     1/0

            # d2['a1f_header'] = a1f_header
            # d2['dmf_header'] = dmf_header

            if j==0:
                d3 = d2.copy()
            else:
                d3 = vstack((d3,d2))
            # 1/0


        out_file = os.path.join(standardized_data_dir, 'FPA_data_%s_SUPER%s_%s_chip%d_%s_%s_%s.fits' % (
                telescope, instr, aperture, chip_id, filter, epoch_isot,
                os.path.basename(f).split('.')[0])).replace(':','-')
        print('Writing %s' % out_file)
        d3.meta['PROPOSID'] = d2['PROPOSID'][0]
        d3.meta['PROGRAM_VISIT'] = '{}_{}'.format(d3.meta['PROPOSID'], d3.meta['DATAFILE'][4:6])
        d3.meta['INSTRUME'] = 'SUPER' + d3.meta['INSTRUME']
        d3.write(out_file, overwrite=True)
