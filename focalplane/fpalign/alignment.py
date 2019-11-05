"""Classes and function to support focal plane alignment of space observatories.

Authors
-------

    Johannes Sahlmann

Use
---

"""

import copy
import os
import pickle
from collections import OrderedDict

from astropy.time import Time
from astropy.table import Table, vstack, hstack
from astropy import units as u
import numpy as np
import pylab as pl

import pysiaf
from pysiaf.utils.tools import get_grid_coordinates
import pystortion

from pystortion.utils import plot_spatial_difference

deg2arcsec = u.deg.to(u.arcsec)



# these are the aperture parameters that define the alignment
alignment_definition_attributes = {'default': ['V3IdlYAngle', 'V2Ref', 'V3Ref'],
                                   'hst_fgs': ['db_tvs_pa_deg', 'db_tvs_v2_arcsec', 'db_tvs_v3_arcsec']}

alignment_parameter_mapping = OrderedDict({'default': {'v3_angle': 'V3IdlYAngle', 'v2_position': 'V2Ref', 'v3_position': 'V3Ref'},
                               'hst_fgs': {'v3_angle': 'db_tvs_pa_deg', 'v2_position': 'db_tvs_v2_arcsec', 'v3_position': 'db_tvs_v3_arcsec'},
                               'unit': {'v3_angle': u.deg, 'v2_position': u.arcsec, 'v3_position': u.arcsec},
                               'default_inverse': {'V3IdlYAngle': 'v3_angle', 'V2Ref':'v2_position', 'V3Ref':'v3_position'},
                                           })


class AlignmentObservation(object):
    """Class for focal plane alignment obervations of Space Telescopes, e.g. HST and JWST

    attributes:
        gaia_catalog : Gaia reference catalog of stars relevant for the observations
        star_catalog : astrometric catalog of stars derived from the observation

    """

    def __init__(self, observatory, instrument):
        self.observatory = observatory
        self.instrument = instrument
        self.fpa_name_seed = '{}_{}'.format(self.observatory, self.instrument)

    def compute_v2v3(self, aperture, V3IdlYAngle_deg=None, V2Ref_arcsec=None, V3Ref_arcsec=None, verbose=False,
                     method='planar_approximation', use_tel_boresight=True, input_coordinates='tangent_plane'):
        """Perform IDL -> V2V3 transformation (tangent and spherical).

        It is assumed that self.star_catalog['x_idl_arcsec'] are planar coordinates.

        """
        if method == 'planar_approximation':
            if V2Ref_arcsec is None:
                V2Ref_arcsec = aperture.V2Ref
            if V3Ref_arcsec is None:
                V3Ref_arcsec = aperture.V3Ref
            if V3IdlYAngle_deg is None:
                V3IdlYAngle_deg = aperture.V3IdlYAngle

            self.star_catalog['v2_tangent_arcsec'], self.star_catalog['v3_tangent_arcsec'] = aperture.idl_to_tel(
                np.array(self.star_catalog['x_idl_arcsec']), np.array(self.star_catalog['y_idl_arcsec']),
                V3IdlYAngle_deg, V2Ref_arcsec, V3Ref_arcsec, method=method, input_coordinates=input_coordinates,
                output_coordinates='tangent_plane')

            self.star_catalog['v2_tangent_deg'] = self.star_catalog['v2_tangent_arcsec'] / deg2arcsec
            self.star_catalog['v3_tangent_deg'] = self.star_catalog['v3_tangent_arcsec'] / deg2arcsec

            # V2V3_tangent_plane -> V2V3_spherical , perform the tangent-plane de-projection of
            # the catalog stars,
            if use_tel_boresight is False:
                # reference point for projection is the local V2/V3 reference point of the
                # aperture OR AN EXTERNALLY SET VALUE
                self.reference_point_deg = np.array([V2Ref_arcsec / deg2arcsec, V3Ref_arcsec / deg2arcsec])
            else:
                self.reference_point_deg = np.array([0., 0.])

            self.star_catalog['v2_spherical_deg'], self.star_catalog[
                'v3_spherical_deg'] = pysiaf.projection.deproject_from_tangent_plane(
                np.array(self.star_catalog['v2_tangent_deg']), np.array(self.star_catalog['v3_tangent_deg']),
                self.reference_point_deg[0], self.reference_point_deg[1])

            if use_tel_boresight is False:
                # subtract reference point to get back into absolute coordinates
                self.star_catalog['v2_spherical_deg'] = self.star_catalog['v2_spherical_deg'] - \
                                                        self.reference_point_deg[0]
                self.star_catalog['v3_spherical_deg'] = self.star_catalog['v3_spherical_deg'] - \
                                                        self.reference_point_deg[1]

            if aperture.InstrName == 'NIRISS':
                print('ATTENTION: special fix for NIRISS')
                self.star_catalog['v2_spherical_deg'] -= 360.

            self.star_catalog['v2_spherical_arcsec'] = self.star_catalog['v2_spherical_deg'] * deg2arcsec
            self.star_catalog['v3_spherical_arcsec'] = self.star_catalog['v3_spherical_deg'] * deg2arcsec

        elif method == 'spherical_transformation':
            table = copy.deepcopy(self.star_catalog)
            table = compute_idl_to_tel_in_table(table, aperture, V3IdlYAngle_deg=V3IdlYAngle_deg,
                                                V2Ref_arcsec=V2Ref_arcsec, V3Ref_arcsec=V3Ref_arcsec, method=method,
                                                use_tel_boresight=use_tel_boresight)
            self.star_catalog = table


class AlignmentObservationCollection(object):
    """Class for an alignment observation collection,
    e.g. from one or several focal plane alignment programs

    2017-10-10  JSA STScI/AURA

    """
    def __init__(self, observations, observatory):
        self.observations = np.array(observations)
        self.observatory = observatory
        self.set_basic_properties()

    def set_basic_properties(self):
        self.n_observations = len(self.observations)
        T = Table()
        for key in self.observations[0].fpa_data.meta.keys():
            value_list = []
            for j in range(self.n_observations):
                if key in self.observations[j].fpa_data.meta.keys():
                    value_list.append(self.observations[j].fpa_data.meta[key])
                else:  # HST FGS case
                    value_list.append(None)
            T[key] = value_list
            # try:
            #     T[key] = [self.observations[j].fpa_data.meta[key] for j in range(
            # self.n_observations)]
            # except KeyError as e:
            #     print('Setting {} to None (error {})'.format(key, e))
            #     T[key] = None

        for key in ['AperName', 'AperType']:
            T[key] = [getattr(self.observations[j].aperture, key) for j in range(self.n_observations)]

        T['number_of_matched_stars'] = [len(self.observations[j].star_catalog_matched) for j in
                                        range(self.n_observations)]

        self.T = T
        self.T['MJD'] = Time(self.T['EPOCH']).mjd

    def delete_observations(self, index):
        """Delete observations specified by index

        :param index:
        :return:
        """
        self.observations = np.delete(self.observations, index)
        self.n_observations = len(self.observations)
        self.T.remove_rows(index)

    def duplicate_observation(self, index):
        """Duplicate one observation."""
        duplicated_observation = copy.deepcopy(self.observations[index])

        self.observations = np.hstack((self.observations, duplicated_observation))
        self.n_observations = len(self.observations)
        self.T.add_row(self.T[index])
        self.T['duplicate'] = np.zeros(len(self.T))
        duplicate_index = len(self.T) - 1
        self.T['duplicate'][duplicate_index] = 1
        return duplicate_index

    def select_observations(self, index):
        """Select observations specified by index."""
        delete_index = np.setdiff1d(np.arange(self.n_observations), index)
        self.delete_observations(delete_index)

    def sort_by(self, column_name):
        """Sort table T and obsrvations by a certain column

        Parameters
        ----------
        column_name

        Returns
        -------

        """

        if column_name not in self.T.colnames:
            raise RuntimeError('Column {} not valid'.format(column_name))

        sorted_index = np.argsort(np.array(self.T[column_name]))
        self.T = self.T[sorted_index]
        self.observations = self.observations[sorted_index]

    def group_by(self, column_name):
        """
        :param column_name:
                attribute table T will be expanded to hold group_id
        :return:
        """

        self.T['group_id'] = np.zeros(self.n_observations).astype(np.int)

        if (self.observatory == 'JWST') and (column_name == 'obs_id'):
            # special obsid that removes the parallel sequence id
            self.T['obsid_special'] = ['{}{}'.format(s[0:16], s[17:26]) for s in self.T['DATAFILE']]
            for jj, key in enumerate(np.array(self.T.group_by('obsid_special').groups.keys)):
                self.T['group_id'][np.where(np.array(self.T['obsid_special'])==key[0])[0]] = jj


        else:
            tmp_group_id = 0
            # for val in np.unique(self.T[column_name])[::-1]:
            for val in np.unique(self.T[column_name]):
                index = np.where(self.T[column_name] == val)[0]
                self.T['group_id'][index] = tmp_group_id
                tmp_group_id += 1

    def generate_attitude_groups(self, threshold_hours=0.3):
        """Create attitude groups of near-contemporaneous camera images to constrain attitude.

        Returns
        -------

        """
        self.T['attitude_group'] = -1
        self.T['attitude_id'] = 'undefined'
        self.sort_by('MJD')
        group_ids = np.unique(np.array(self.T['group_id']))

        for group_id in group_ids:
            if self.observatory == 'HST':
                # exclude SUPERFGS
                valid_index = np.where((self.T['group_id'] == group_id) & (self.T['INSTRUME'] != 'SUPERFGS'))[0]
            else:
                valid_index = np.where((self.T['group_id'] == group_id))[0]

            # threshold to separate attitude groups
            break_index = np.where(np.diff(self.T['MJD'][valid_index] * 24) > threshold_hours)[0] + 1
            if not break_index:
                # only one group
                self.T['attitude_group'][valid_index] = 0
            else:
                for index in break_index:
                    self.T['attitude_group'][valid_index[0:index]] = 0
                    self.T['attitude_group'][valid_index[index:]] = 1

            for i in valid_index:
                self.T['attitude_id'][i] = '{}-{}'.format(self.T['group_id'][i], self.T['attitude_group'][i])

    def assign_alignment_reference_aperture(self, reference_aperture_name):
        """Given a grouped obs collection, the alignment reference aperture is identified by the
        reference_aperture_name.
        :param reference_aperture_name:
        :return:
        """

        self.T['alignment_reference'] = np.zeros(self.n_observations).astype(np.int)
        for group_id in np.unique(self.T['group_id']):
            # select only first element of rows matching criteria (alignment reference has to be
            # unique for every group)
            index = np.where((self.T['group_id'] == group_id) & (self.T['AperName'] == reference_aperture_name))[0]
            if type(index) == list:
                index = index[0]
            self.T['alignment_reference'][index] = 1


def apply_focal_plane_calibration(obs_collection, apertures_to_calibrate, calibrated_data,
                                  verbose=True, field_selection='calibrated',
                                  calibrated_obs_collection=None):
    """Modify the alignment parameters of the used apertures to reflect the result of a previous alignment run.

    Set the alignment parameters for the apertures_to_calibrate as determined in an independent calibration run.

    Parameters
    ----------
    obs_collection
    apertures_to_calibrate
    calibrated_data

    Returns
    -------

    """

    for tmp_j, tmp_aperture_name in enumerate(obs_collection.T['AperName']):
        if tmp_aperture_name in apertures_to_calibrate:
            calibration_index = np.where((np.int(obs_collection.T['PROPOSID'][tmp_j]) == np.array(calibrated_data['PROPOSID'])) &
                                         (np.array(calibrated_data['AperName']) == tmp_aperture_name) &
                                         (np.int(obs_collection.T['EPOCHNUMBER'][tmp_j]) == np.array(calibrated_data['EPOCHNUMBER']).astype(np.int)))[0]
            print('+++++++++++ APPLYING CALIBRATION +++++++++++')
            if verbose:
                print('Applying calibration as mean of')
                calibrated_data['DATE-OBS PROGRAM_VISIT APERTURE CHIP {0}_V2Ref ' \
                                    '{0}_V3Ref {0}_V3IdlYAngle'.format(field_selection).split()][calibration_index].pprint()

            alignment_parameter_set = obs_collection.T['align_params'][calibration_index[0]]  # 'default' or 'hst_fgs'
            for key, attribute in alignment_parameter_mapping[alignment_parameter_set].items():
                mean_value = np.mean(calibrated_data['{}_{}'.format(field_selection, attribute)][calibration_index])
                print('Setting {} from {:2.3f} to {:2.3f} (average of {} samples) for alignment_reference_aperture '
                      '{}'.format(attribute, getattr(obs_collection.observations[tmp_j].aperture, attribute), mean_value,
                                  len(calibration_index), tmp_aperture_name))
                setattr(obs_collection.observations[tmp_j].aperture, attribute, mean_value)

            # transfer distortion coefficients to allow skipping the alignment during the next attitude determination
            if calibrated_obs_collection is not None:
                subset_index = 0 # take the deep or shallow exposure
                assert calibrated_obs_collection.T['AperName'][calibration_index[subset_index]] == obs_collection.T['AperName'][tmp_j]
                obs_collection.observations[tmp_j].distortion_dict = calibrated_obs_collection.observations[calibration_index[subset_index]].distortion_dict

    return obs_collection


def compute_idl_to_tel_in_table(input_table, aperture, V3IdlYAngle_deg=None, V2Ref_arcsec=None, V3Ref_arcsec=None,
                                verbose=False, method='planar_approximation', distortion_correction=None,
                                use_tel_boresight=True, input_coordinates=None):
    """Perform IDL -> V2V3_tangent transformation and deprojection to V2V3_spherical.
    The input table must have  columns named 'x_idl_arcsec' and 'y_idl_arcsec'.

    Parameters
    ----------
    input_table
    aperture
    V3IdlYAngle_deg
    V2Ref_arcsec
    V3Ref_arcsec
    verbose
    method
    distortion_correction : dict
        Allows correction for distortion before determining alignment parameters
    use_tel_boresight : bool
        if True, the V2=0, V3=0 origin of the tel coordinate system us used as deprojection point
        if False, the aperture's V2Ref, V3Ref parameters are used


    Returns
    -------

    """
    table = copy.deepcopy(input_table)
    for colname in 'v2_tangent_arcsec v3_tangent_arcsec v2_spherical_deg v3_spherical_deg v2_spherical_arcsec v3_spherical_arcsec v2_tangent_deg v3_tangent_deg'.split():
        if colname in table.colnames:
            table.remove_column(colname)

    if 'x_idl_arcsec' not in table.colnames:
        raise ValueError('IDL coordinates not in table or wrong column name')

    if method == 'planar_approximation':
        table['v2_tangent_arcsec'], table['v3_tangent_arcsec'] = aperture.idl_to_tel(np.array(table['x_idl_arcsec']),
            np.array(table['y_idl_arcsec']), V3IdlYAngle_deg=V3IdlYAngle_deg, V2Ref_arcsec=V2Ref_arcsec, V3Ref_arcsec=V3Ref_arcsec, method=method, input_coordinates='tangent_plane', output_coordinates='tangent_plane')

        if distortion_correction is not None:
            # correct for distortion
            fieldname_dict = distortion_correction['fieldname_dict']
            x_original = np.array(table[fieldname_dict['star_catalog']['position_1']])  # == 'v2_tangent_arcsec'
            y_original = np.array(table[fieldname_dict['star_catalog']['position_2']])
            table['v2_tangent_arcsec'], table['v3_tangent_arcsec'] = distortion_correction[
                'coefficients'].apply_polynomial_transformation(distortion_correction['evaluation_frame_number'],
                x_original, y_original)
            if 0:
                distortion_correction['coefficients'].plotDistortion(distortion_correction['evaluation_frame_number'],
                    '', '', distortion_correction['reference_point_for_projection'], save_plot=0)

        table['v2_tangent_deg'] = table['v2_tangent_arcsec'] / deg2arcsec
        table['v3_tangent_deg'] = table['v3_tangent_arcsec'] / deg2arcsec

        # V2V3_tangent_plane -> V2V3_spherical , perform the tangent-plane de-projection of the
        # catalog stars,
        # reference point for projection is the local V2/V3 reference point of the aperture the
        # tel boresight
        if use_tel_boresight is False:
            reference_point_deg = np.array([aperture.V2Ref / deg2arcsec, aperture.V3Ref / deg2arcsec])
        else:
            reference_point_deg = np.array([0., 0.])

        if verbose:
            print('Reference point position {0:3.8f} / {1:3.8f} arcsec'.format(
                reference_point_deg[0] * u.deg.to(u.arcsec), reference_point_deg[1] * u.deg.to(u.arcsec)))

        table['v2_spherical_deg'], table['v3_spherical_deg'] = pysiaf.projection.deproject_from_tangent_plane(
            np.array(table['v2_tangent_deg']), np.array(table['v3_tangent_deg']), reference_point_deg[0],
            reference_point_deg[1])

        if use_tel_boresight is False:
            # subtract reference point to get back into absolute coordinates
            table['v2_spherical_deg'] = table['v2_spherical_deg'] - reference_point_deg[0]
            table['v3_spherical_deg'] = table['v3_spherical_deg'] - reference_point_deg[1]

        table['v2_spherical_deg'][table['v2_spherical_deg'].data > 180.] -= 360.

    elif method == 'spherical_transformation':
        v2_spherical_arcsec, v3_spherical_arcsec = aperture.idl_to_tel(np.array(table['x_idl_arcsec']),
            np.array(table['y_idl_arcsec']), V3IdlYAngle_deg=V3IdlYAngle_deg, V2Ref_arcsec=V2Ref_arcsec,
            V3Ref_arcsec=V3Ref_arcsec, method='spherical_transformation', input_coordinates='tangent_plane', output_coordinates='spherical')

        v2_spherical_deg, v3_spherical_deg = v2_spherical_arcsec / 3600., v3_spherical_arcsec / 3600.
        table['v2_spherical_deg'], table['v3_spherical_deg'] = v2_spherical_deg, v3_spherical_deg

        # tangent plane projection using boresight
        table['v2_tangent_deg'], table['v3_tangent_deg'] = pysiaf.projection.project_to_tangent_plane(v2_spherical_deg, v3_spherical_deg,
                                                                                                      0., 0.)
        table['v2_tangent_arcsec'], table['v3_tangent_arcsec'] = table['v2_tangent_deg'] * 3600., table[
            'v3_tangent_deg'] * 3600.

        if distortion_correction is not None:
            # correct for distortion
            fieldname_dict = distortion_correction['fieldname_dict']
            x_original = np.array(table[fieldname_dict['star_catalog']['position_1']])  # == 'v2_tangent_arcsec'
            y_original = np.array(table[fieldname_dict['star_catalog']['position_2']])
            table['v2_tangent_arcsec'], table['v3_tangent_arcsec'] = distortion_correction[
                'coefficients'].apply_polynomial_transformation(distortion_correction['evaluation_frame_number'],
                x_original, y_original)
            table['v2_tangent_deg'] = table['v2_tangent_arcsec'] / deg2arcsec
            table['v3_tangent_deg'] = table['v3_tangent_arcsec'] / deg2arcsec

        table['v2_spherical_deg'][table['v2_spherical_deg'].data > 180.] -= 360.

    elif method == 'spherical':
        if aperture.AperName in ['FGS1', 'FGS2', 'FGS3']:
            input_coordinates = 'cartesian'
        else:
            input_coordinates = 'polar'

        table['v2_spherical_arcsec'], table['v3_spherical_arcsec'] = aperture.idl_to_tel(np.array(table['x_idl_arcsec']),
        np.array(table['y_idl_arcsec']), V3IdlYAngle_deg=V3IdlYAngle_deg, V2Ref_arcsec=V2Ref_arcsec,
        V3Ref_arcsec=V3Ref_arcsec, method=method, input_coordinates=input_coordinates, output_coordinates='polar')

        table['v2_spherical_deg'], table['v3_spherical_deg'] = table['v2_spherical_arcsec']/3600., table['v3_spherical_arcsec']/3600.

        if distortion_correction is not None:
            # correct for distortion
            fieldname_dict = distortion_correction['fieldname_dict']
            v2_name = fieldname_dict['star_catalog']['position_1'] # == 'v2_spherical_arcsec'
            v3_name = fieldname_dict['star_catalog']['position_2'] # == 'v3_spherical_arcsec'
            x_original = np.array(table[v2_name])
            y_original = np.array(table[v3_name])

            table[v2_name], table[v3_name] = distortion_correction['coefficients'].apply_polynomial_transformation(
                distortion_correction['evaluation_frame_number'], x_original, y_original)

            table['v2_spherical_deg'] = table[v2_name] / deg2arcsec
            table['v3_spherical_deg'] = table[v3_name] / deg2arcsec
        table['v2_spherical_deg'][table['v2_spherical_deg'].data > 180.] -= 360.

    if 'v2_spherical_arcsec' not in table.colnames:
        table['v2_spherical_arcsec'], table['v3_spherical_arcsec'] = table['v2_spherical_deg']*3600.,  table['v3_spherical_deg']*3600

    return table


def compute_sky_to_tel_in_table(table, attitude, aperture, verbose=False, use_tel_boresight=True):
    """Perform Ra/Dec -> V2V3_spherical transformation and projection to V2V3_tangent.

    The input table must have  columns named 'ra' and 'dec'.

    :param table:
    :param attitude:
    :param aperture:
    :param verbose:
    :return:

    """

    if 'ra' not in table.colnames:
        raise ValueError('sky coordinates not in table or wrong column name')

    table['v2_spherical_arcsec'], table['v3_spherical_arcsec'] = pysiaf.rotations.getv2v3(attitude,
        np.array(table['ra']), np.array(table['dec']))

    if use_tel_boresight is False:
        # use V2/V3 REF of aperture as reference point for tangent-plane projection of catalog
        reference_point_deg = np.array([aperture.V2Ref / deg2arcsec, aperture.V3Ref / deg2arcsec])
    else:
        # use boresight, V2=0, V3=0
        reference_point_deg = np.array([0., 0.])

    if verbose:
        print('Reference point position RA/Dec {0:3.8f} / {1:3.8f}'.format(reference_point_deg[0] * u.deg.to(u.arcsec),
                                                                           reference_point_deg[1] * u.deg.to(u.arcsec)))

    table['v2_tangent_deg'], table['v3_tangent_deg'] = pysiaf.projection.project_to_tangent_plane(
        np.array(table['v2_spherical_arcsec'] / deg2arcsec), np.array(table['v3_spherical_arcsec'] / deg2arcsec),
        reference_point_deg[0], reference_point_deg[1])

    if use_tel_boresight is False:
        # add back the projection reference point to get absolute V2V3 coordinates
        table['v2_tangent_deg'] = table['v2_tangent_deg'] + reference_point_deg[0]
        table['v3_tangent_deg'] = table['v3_tangent_deg'] + reference_point_deg[1]

    table['v2_tangent_arcsec'] = table['v2_tangent_deg'] * deg2arcsec
    table['v3_tangent_arcsec'] = table['v3_tangent_deg'] * deg2arcsec

    return table


def compute_tel_to_idl_in_table(table, aperture, use_tel_boresight=True, method='planar_approximation'):
    """perform tangent plane projection from V2V3_spherical V2V3_tangent
    then transform to to Ideal

    The input table must have  columns named 'v2_spherical_arcsec' and 'v3_spherical_arcsec'

    :param table:
    :param aperture:
    :param V3IdlYAngle_deg:
    :param V2Ref_arcsec:
    :param V3Ref_arcsec:
    :param verbose:
    :return:
    """

    if 'v2_spherical_arcsec' not in table.colnames:
        raise ValueError('Tel coordinates not in table or wrong column name')

    if 'v2_spherical_deg' not in table.colnames:
        table['v2_spherical_deg'] = table['v2_spherical_arcsec'] / deg2arcsec
        table['v3_spherical_deg'] = table['v3_spherical_arcsec'] / deg2arcsec

    if use_tel_boresight is False:
        reference_point_deg = np.array([aperture.V2Ref / deg2arcsec, aperture.V3Ref / deg2arcsec])
    else:
        reference_point_deg = np.array([0., 0.])

    if method == 'planar_approximation':
        table['v2_tangent_deg'], table['v3_tangent_deg'] = pysiaf.projection.project_to_tangent_plane(np.array(table['v2_spherical_deg']), np.array(table['v3_spherical_deg']), reference_point_deg[0], reference_point_deg[1])

        table['v2_tangent_arcsec'] = table['v2_tangent_deg'] * deg2arcsec
        table['v3_tangent_arcsec'] = table['v3_tangent_deg'] * deg2arcsec

        table['x_idl_arcsec'], table['y_idl_arcsec'] = aperture.tel_to_idl(table['v2_tangent_arcsec'],
                                                                           table['v3_tangent_arcsec'],
                                                                           method=method,
                                                                           output_coordinates='tangent_plane')
    elif method == 'spherical_transformation':
        table['x_idl_arcsec'], table['y_idl_arcsec'] = aperture.tel_to_idl(table['v2_spherical_arcsec'],
                                                                           table['v3_spherical_arcsec'],
                                                                           method=method,
                                                                           output_coordinates='tangent_plane')

    return table


def determine_aperture_error(obs, reference_obs, obs_index, reference_observation_index,
                             alignment_reference_aperture=None, maximum_number_of_iterations=100, attenuation_factor=0.9,
                             fractional_threshold_for_iterations=0.01, verbose=False, k=4, plot_residuals=False,
                             reference_frame_number=0, evaluation_frame_number=1,
                             reference_point_setting='auto', rotation_name='Rotation in Y',
                             plot_dir=None, distortion_correction=None, idl_tel_method='spherical',
                             use_tel_boresight=False, parameters={}):
    """Use a polynomial fit to determine the aperture error.

    This returns corrected values of V2Ref, V3Ref, V3IdlYAngle of the obs.aperture.
    Iterative polynomial fit to determine corrections to V2Ref, V3Ref, V3IdlYAngle (attitude remains
    fixed).

    Parameters
    ----------
    obs : AlignmentObservation instance
        The observation to be processed
    reference_obs : AlignmentObservation instance
        The alignment reference observation. This is the observation that needs to be processed
        first.
    obs_index : int
        index of obs in AlignmentObservationCollection
    reference_observation_index : int
        index of reference_obs in AlignmentObservationCollection
    maximum_number_of_iterations : int
        When to stop iterating.
    attenuation_factor : float
        Gain applied to interative corrections.
    fractional_threshold_for_iterations : float
        determines when the iteration has converged by comparing the uncertainty of the correction
        to the correction amplitude.
        iteration stops when np.abs(correction) < fractional_threshold_for_iterations * uncertainty
    verbose : bool
        Verbosity
    k : int
        Parameter that defines the polynomial degree for distortion fitting
    plot_residuals : bool
        Whether to make residual figures.
    reference_frame_number : int
    evaluation_frame_number : int
    reference_point : numpy array
    rotation_name : str
    plot_dir : str
    distortion_correction : dict
        Allows correction for distortion before determining alignment parameters

    Returns
    -------

    """
    iteration_number = 0
    converged = False

    if (obs.aperture.observatory == 'HST') & ('FGS' in obs.aperture.AperName):
        hst_fgs_case = True
    else:
        hst_fgs_case = False

    if reference_point_setting == 'auto':
        if hst_fgs_case:
            use_fiducial_as_reference_point = parameters['use_hst_fgs_fiducial_as_reference_point']
        else:
            use_fiducial_as_reference_point = True

        # this shifts the origin (center of rotation) for the distortion fit from
        # v2,v3=0,0 to the fiducial point of the aperture to mitigate correlation
        # between offsets and rotation
        if use_fiducial_as_reference_point:
            reference_point = np.array(
                [[obs.aperture.V2Ref_original, obs.aperture.V3Ref_original], [obs.aperture.V2Ref_original, obs.aperture.V3Ref_original]])
        else:
            # for HST FGS   sequence is [Xref, Yref], [Xref,Yref]
            reference_point = np.array([[0., 0.], [0., 0.]])
    else:
        reference_point = reference_point_setting

    if hst_fgs_case:
        print('determine_aperture_error: Entering special mode for HST-FGS')
        # V3IdlYAngle and V2/3Ref attributes are overloaded with parameters that define the
        # HST-FGS TVS matrix = alignment matrix

        obs.aperture.V3IdlYAngle = obs.aperture.db_tvs_pa_deg
        obs.aperture.V2Ref = obs.aperture.db_tvs_v2_arcsec
        obs.aperture.V3Ref = obs.aperture.db_tvs_v3_arcsec

    for j in np.arange(maximum_number_of_iterations):
        if verbose:
            print('{}\nIteration {}'.format('~' * 10, iteration_number))
        # initialize alignment parameters
        if iteration_number == 0:
            if obs_index == reference_observation_index:
                # Processing reference observation. Setting alignment parameters directly.
                V3IdlYAngle_deg = obs.aperture.V3IdlYAngle
                V2Ref_arcsec = obs.aperture.V2Ref
                V3Ref_arcsec = obs.aperture.V3Ref
            else:
                # Setting alignment parameters relative to reference observation.
                # use  and calibrate relative V2V3 parameters between actual and reference aperture
                relative_V3IdlYAngle_deg = obs.aperture.V3IdlYAngle - reference_obs.aperture.V3IdlYAngle
                relative_V2Ref_arcsec = obs.aperture.V2Ref - reference_obs.aperture.V2Ref
                relative_V3Ref_arcsec = obs.aperture.V3Ref - reference_obs.aperture.V3Ref

                V3IdlYAngle_deg = reference_obs.aperture.V3IdlYAngle_corrected + relative_V3IdlYAngle_deg
                V2Ref_arcsec = reference_obs.aperture.V2Ref_corrected + relative_V2Ref_arcsec
                V3Ref_arcsec = reference_obs.aperture.V3Ref_corrected + relative_V3Ref_arcsec
                if verbose:
                    print(relative_V2Ref_arcsec, '=', obs.aperture.V2Ref, reference_obs.aperture.V2Ref)
                    print(V2Ref_arcsec, '=', reference_obs.aperture.V2Ref_corrected, relative_V2Ref_arcsec)
        else:
            # use parameters determined in previous iteration to determine correction terms
            if hasattr(obs.aperture, '_fgs_use_rearranged_alignment_parameters') is False:
                obs.aperture._fgs_use_rearranged_alignment_parameters = None
            if hst_fgs_case & (obs.aperture._fgs_use_rearranged_alignment_parameters is False):
                distortion_rotation_deg, distortion_shift_in_X, distortion_shift_in_Y, converged, \
                sigma_distortion_rotation_deg, sigma_distortion_shift_in_X, sigma_distortion_shift_in_Y = \
                    determine_corrections(lazAC, fractional_threshold_for_iterations, rotation_name, verbose=True)

                # empirically determined relationships and signs for FGS TVS parameters, i.e.
                # how is a shift in v2 reflected in a change of a TVS parameter?
                if obs.aperture.AperName == 'FGS3':
                    current_rotation_deg = -1 * copy.deepcopy(distortion_shift_in_X) * u.arcsec.to(u.deg)
                    current_shift_in_X = +1* copy.deepcopy(distortion_rotation_deg) * u.deg.to(u.arcsec)
                    current_shift_in_Y = +1* copy.deepcopy(distortion_shift_in_Y)
                    sigma_current_rotation_deg = sigma_distortion_shift_in_X * u.arcsec.to(u.deg)
                    sigma_current_shift_in_X = sigma_distortion_rotation_deg * u.deg.to(u.arcsec)
                    sigma_current_shift_in_Y = sigma_distortion_shift_in_Y
                elif obs.aperture.AperName == 'FGS1':
                    current_rotation_deg = +1 * copy.deepcopy(distortion_shift_in_X) * u.arcsec.to(u.deg)
                    current_shift_in_X = +1* copy.deepcopy(distortion_rotation_deg) * u.deg.to(u.arcsec)
                    current_shift_in_Y = -1* copy.deepcopy(distortion_shift_in_Y)
                    sigma_current_rotation_deg = sigma_distortion_shift_in_X * u.arcsec.to(u.deg)
                    sigma_current_shift_in_X = sigma_distortion_rotation_deg * u.deg.to(u.arcsec)
                    sigma_current_shift_in_Y = sigma_distortion_shift_in_Y
                elif obs.aperture.AperName == 'FGS2':
                    current_rotation_deg = -1 * copy.deepcopy(distortion_shift_in_Y) * u.arcsec.to(u.deg)
                    current_shift_in_X = +1* copy.deepcopy(distortion_rotation_deg) * u.deg.to(u.arcsec)
                    current_shift_in_Y = -1* copy.deepcopy(distortion_shift_in_X)
                    sigma_current_rotation_deg = sigma_distortion_shift_in_Y * u.arcsec.to(u.deg)
                    sigma_current_shift_in_X = sigma_distortion_rotation_deg * u.deg.to(u.arcsec)
                    sigma_current_shift_in_Y = sigma_distortion_shift_in_X

            else:
                # not HST-FGS case
                current_rotation_deg, current_shift_in_X, current_shift_in_Y, converged, sigma_current_rotation_deg, \
                sigma_current_shift_in_X, sigma_current_shift_in_Y = determine_corrections(lazAC, fractional_threshold_for_iterations, rotation_name)

            # compute correction terms
            correction_V3IdlYAngle_deg = attenuation_factor * current_rotation_deg
            correction_V2Ref_arcsec = attenuation_factor * current_shift_in_X
            correction_V3Ref_arcsec = attenuation_factor * current_shift_in_Y

            # apply correction terms
            V2Ref_arcsec += correction_V2Ref_arcsec
            V3Ref_arcsec += correction_V3Ref_arcsec
            V3IdlYAngle_deg += correction_V3IdlYAngle_deg

            if verbose:
                print('correction_V2Ref_arcsec = {}'.format(correction_V2Ref_arcsec))
                print('correction_V3Ref_arcsec = {}'.format(correction_V3Ref_arcsec))
                print('correction_V3IdlYAngle_deg = {}'.format(correction_V3IdlYAngle_deg))

        if converged:
            print('Aperture correction: Iterations converged after {} steps'.format(iteration_number))
            break

        if verbose:
            print('V2Ref_arcsec = {}'.format(V2Ref_arcsec))
            print('V3Ref_arcsec = {}'.format(V3Ref_arcsec))
            print('V3IdlYAngle_deg = {}'.format(V3IdlYAngle_deg))


        # prepare data for distortion fit
        obs.star_catalog_matched = compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                               V3IdlYAngle_deg=V3IdlYAngle_deg,
                                                               V2Ref_arcsec=V2Ref_arcsec,
                                                               V3Ref_arcsec=V3Ref_arcsec,
                                                               distortion_correction=distortion_correction,
                                                               method=idl_tel_method,
                                                               use_tel_boresight=use_tel_boresight)

        mp = pystortion.distortion.prepare_multi_epoch_astrometry(obs.star_catalog_matched, obs.gaia_catalog_matched,
                                                                  fieldname_dict=obs.fieldname_dict)

        # perform distortion fit
        lazAC, index_masked_stars = pystortion.distortion.fit_distortion_general(mp, k,
                                                                                 eliminate_omc_outliers_iteratively=1,
                                                                                 outlier_rejection_level_sigma=3.,
                                                                                 reference_frame_number=reference_frame_number,
                                                                                 evaluation_frame_number=evaluation_frame_number,
                                                                                 reference_point=reference_point,
                                                                                 verbose=verbose)

        # convert to easily read parameters
        lazAC.human_readable_solution_parameters = pystortion.distortion.compute_rot_scale_skew(lazAC, i=evaluation_frame_number)

        iteration_number += 1

    # after iteration converged or reached maximum number of iterations
    if iteration_number == (maximum_number_of_iterations - 1):
        raise RuntimeWarning('Iterative aperture correction did not converge')
    else:
        if plot_residuals:
            # show the residuals of the last distortion fit
            xy_unit = u.arcsec
            xy_scale = u.arcsecond.to(xy_unit)
            xy_unitStr = xy_unit.to_string()
            name_seed = obs.fpa_name_seed + '_k{:d}'.format(k)
            # lazAC.display_results(evaluation_frame_number=evaluation_frame_number,
            # scale_factor_for_residuals=1., displayCorrelations=0, nformat='f')
            lazAC.plotResiduals(evaluation_frame_number, plot_dir, name_seed,
                                omc_scale=u.arcsecond.to(u.milliarcsecond), save_plot=True, omc_unit='mas',
                                xy_scale=xy_scale, xy_unit=xy_unitStr, bins=10, title=obs.aperture.AperName)

        # store the uncertainties in the determined parameters by setting attributes of obs,
        attribute_mapping = {'rotation_deg': rotation_name, 'shift_in_X': 'Shift in X', 'shift_in_Y': 'Shift in Y', }
        values = lazAC.human_readable_solution_parameters['values']
        names = lazAC.human_readable_solution_parameters['names'].tolist()
        for key in attribute_mapping:
            setattr(obs, 'sigma_current_{}'.format(key), values[names.index(attribute_mapping[key])][1])

        if index_masked_stars is None:
            obs.number_of_used_stars_for_aperture_correction = copy.deepcopy(obs.number_of_matched_stars)
        else:
            obs.number_of_used_stars_for_aperture_correction = copy.deepcopy(obs.number_of_matched_stars) - len(
                index_masked_stars)

        if hst_fgs_case:
            obs.aperture.corrected_tvs_v2_arcsec = V2Ref_arcsec
            obs.aperture.corrected_tvs_v3_arcsec = V3Ref_arcsec
            obs.aperture.corrected_tvs_pa_deg = V3IdlYAngle_deg

            # obs.aperture.db_tvs_v2_arcsec = V2Ref_arcsec
            # obs.aperture.db_tvs_v3_arcsec = V3Ref_arcsec
            # obs.aperture.db_tvs_pa_deg = V3IdlYAngle_deg
            obs.aperture.db_tvs_v2_arcsec_corrected = V2Ref_arcsec
            obs.aperture.db_tvs_v3_arcsec_corrected = V3Ref_arcsec
            obs.aperture.db_tvs_pa_deg_corrected = V3IdlYAngle_deg

            obs.updated_tvs_matrix = obs.aperture.compute_tvs_matrix(
                v2_arcsec=obs.aperture.corrected_tvs_v2_arcsec,
                v3_arcsec=obs.aperture.corrected_tvs_v3_arcsec,
                pa_deg=obs.aperture.corrected_tvs_pa_deg)

            obs.aperture.corrected_tvs = obs.aperture._tvs_parameters(obs.updated_tvs_matrix)[3]
            obs.aperture.tvs_corrected = obs.aperture._tvs_parameters(obs.updated_tvs_matrix)[3]

            if 0:
                obs.aperture.set_tel_reference_point()  # this sets V2Ref,V3Ref back to the SIAF values
                idl_x_ref_arcsec, idl_y_ref_arcsec = obs.aperture.tel_to_idl(obs.aperture.V2Ref, obs.aperture.V3Ref)

                print(obs.aperture)
                print(obs.aperture.V2Ref, obs.aperture.V3Ref)
                tvs_parameters = obs.aperture._tvs_parameters(tvs=obs.aperture.tvs_corrected)
                # original_fiducial_position (not the same as SIAF value)
                obs.aperture.V2Ref, obs.aperture.V3Ref = obs.aperture.idl_to_tel(idl_x_ref_arcsec, idl_y_ref_arcsec)
                print(obs.aperture.V2Ref, obs.aperture.V3Ref)

                # updated fiducial position
                obs.aperture.V2Ref_corrected, obs.aperture.V3Ref_corrected = \
                    obs.aperture.idl_to_tel(idl_x_ref_arcsec, idl_y_ref_arcsec,
                                        V3IdlYAngle_deg=tvs_parameters[2],
                                        V2Ref_arcsec=tvs_parameters[0],
                                        V3Ref_arcsec=tvs_parameters[1])
                # obs.aperture.corrected_V2Ref, obs.aperture.corrected_V3Ref = \
                obs.aperture.corrected_V2Ref, obs.aperture.corrected_V3Ref = obs.aperture.V2Ref_corrected, obs.aperture.V3Ref_corrected
                print(obs.aperture.V2Ref_corrected, obs.aperture.V3Ref_corrected)
            else:
                obs.aperture.set_tel_reference_point()  # this sets V2Ref,V3Ref back to the SIAF values
                # V2Ref_corrected,V3Ref_corrected are set according to the idl_to_tel transformation of
                # aperture.idl_x_ref_arcsec,aperture.idl_y_ref_arcsec using the aperture.corrected_tvs
                # V3IdlYAngle and V3IdlYAngle_corrected are set back to original SIAF value




            obs.translation = None
            # show_fgs_pickle_correction(obs.aperture, obs.aperture.corrected_tvs, factor=1000,
            #                            show_original=True)

            if verbose:
                db_tvs_v2_arcsec, db_tvs_v3_arcsec, db_tvs_pa_deg, tvs = obs.aperture._tvs_parameters()
                print('Derived corrections to TVS parameters:')
                print('delta tvs_v2: {} arcsec'.format(
                    db_tvs_v2_arcsec - obs.aperture.corrected_tvs_v2_arcsec))
                print('delta tvs_v3: {} arcsec'.format(
                    db_tvs_v3_arcsec - obs.aperture.corrected_tvs_v3_arcsec))
                print('delta tvs_pa: {} deg'.format(
                    db_tvs_pa_deg - obs.aperture.corrected_tvs_pa_deg))

        else:
            obs.aperture.V3IdlYAngle_corrected = V3IdlYAngle_deg
            obs.aperture.V2Ref_corrected = V2Ref_arcsec
            obs.aperture.V3Ref_corrected = V3Ref_arcsec

        obs.lazAC = lazAC

        for attribute in ['V3IdlYAngle', 'V2Ref', 'V3Ref']:
            original = getattr(obs.aperture, '{}'.format(attribute))
            corrected = getattr(obs.aperture, '{}_corrected'.format(attribute))
            print('Aperture correction: {} original: {:3.4f} \t corrected {:3.4f} \t difference {'
                  ':3.4f}'.format(attribute, original, corrected, corrected - original))

        distortion_dict = {'fieldname_dict'         : obs.fieldname_dict,
                           'coefficients'  : lazAC,
                           'evaluation_frame_number': parameters['evaluation_frame_number']}

        obs.distortion_dict = distortion_dict

    return


def determine_attitude(obs_set, parameters):
    """Return a refined attitude estimate.

    Iterative polynomial fit to determine corrections to RA_V1, Dec_V1, and PA_V3,
    while aperture alignment parameters (V2Ref,V3Ref, V3IdlYangle) remain fixed.
    If obs_set contains several apertures, the attitude-defining aperture is kept
    fixed and temporary alignments are made to the remaining apertures.

    Parameters
    ----------
    obs_set : AlignmentObservationCollection
        A collection of observations to use for attitude determination.
    parameters : dict
        Dictionary of configuration parameters, contains:
        - attitude_defining_aperture : str
        - maximum_number_of_iterations : int
        - attenuation_factor : float
        - fractional_threshold_for_iterations : float
            determines when the iteration has converged by comparing the uncertainty of the
            correction
            to the correction amplitude.
            iteration stops when np.abs(correction) < fractional_threshold_for_iterations *
            uncertainty
        - verbose : bool
        - k_attitude_determination : int
        - reference_frame_number
        - evaluation_frame_number
        - reference_point
        - rotation_name
        - use_v1_pointing : bool
            if True, the attitude is determined at the V-frame origin
            if False, the attitude is determined at the aperture's fiducial point


    Returns
    -------
    attitude_dict : dict
        Dictionary holding results of attitude determination

    """
    perform_alignment_update = parameters['perform_temporary_alignment_update']
    perform_distortion_correction = parameters['perform_temporary_distortion_correction']

    if perform_alignment_update is False:
        perform_distortion_correction = False

    obs_set = copy.deepcopy(obs_set)
    aperture_names = np.array(obs_set.T['AperName'])
    unique_aperture_names = np.unique(aperture_names)
    if len(unique_aperture_names) != obs_set.n_observations:
        raise RuntimeError('Attitude set has duplicate apertures.')

    k = parameters['k_attitude_determination']
    verbose = parameters['verbose']
    use_tel_boresight = parameters['use_tel_boresight']

    attenuation_factor = parameters['attenuation_factor']
    maximum_number_of_iterations = parameters['maximum_number_of_iterations']
    attitude_defining_aperture_name = parameters['attitude_defining_aperture_name']
    attitude_defining_aperture_index = np.where(aperture_names == attitude_defining_aperture_name)[0][0]
    attitude_defining_obs = obs_set.observations[attitude_defining_aperture_index]
    attitude_defining_aperture = attitude_defining_obs.aperture

    print('='*50)
    print('determine_attitude: received obs_set with {} observations'.format(obs_set.n_observations))
    print('determine_attitude: attitude-defining aperture is {}'.format(attitude_defining_aperture_name))

    # make sure that attitude_defining_aperture is processed first
    # order observations by increasing separation from attitude_defining_aperture
    separation_tel_arcsec = np.zeros(obs_set.n_observations)
    for j in range(obs_set.n_observations):
        aperture = obs_set.observations[j].aperture
        separation_tel_arcsec[j] = np.sqrt((attitude_defining_aperture.V2Ref - aperture.V2Ref) ** 2 + (
            attitude_defining_aperture.V3Ref - aperture.V3Ref) ** 2)
    attitude_order_index = np.argsort(separation_tel_arcsec)

    if obs_set.observatory == 'JWST':
        ra_pointing_keyword = 'pointing_ra_v1'
        dec_pointing_keyword = 'pointing_dec_v1'
        pa_pointing_keyword = 'pointing_pa_v3'
        V2Ref = 0.
        V3Ref = 0.

    elif obs_set.observatory == 'HST':
        if parameters['use_v1_pointing']:
            ra_pointing_keyword = 'RA_V1'
            dec_pointing_keyword = 'DEC_V1'
            pa_pointing_keyword = 'PA_V3'
            V2Ref = 0.
            V3Ref = 0.
        else:
            ra_pointing_keyword = 'RA_APER'
            dec_pointing_keyword = 'DEC_APER'
            pa_pointing_keyword = 'PA_APER'
            V2Ref = attitude_defining_aperture.V2Ref
            V3Ref = attitude_defining_aperture.V3Ref

    # initial attitude from header of one observation
    ra_attitude_deg = attitude_defining_obs.fpa_data.meta[ra_pointing_keyword]
    dec_attitude_deg = attitude_defining_obs.fpa_data.meta[dec_pointing_keyword]
    pa_attitude_deg = attitude_defining_obs.fpa_data.meta[pa_pointing_keyword]

    initial_ra_attitude_deg = copy.deepcopy(ra_attitude_deg)
    initial_pa_attitude_deg = copy.deepcopy(pa_attitude_deg)
    initial_dec_attitude_deg = copy.deepcopy(dec_attitude_deg)

    fieldname_dict = attitude_defining_obs.fieldname_dict

    attitude = pysiaf.rotations.attitude(V2Ref, V3Ref, ra_attitude_deg, dec_attitude_deg, pa_attitude_deg)

    # show offset between measured positions and reference positions using initial attitude
    if 0:
        plot_aperture_names = ['FGS1', 'FGS2', 'FGS3', 'JWFCFIX', 'JWFC1FIX', 'JWFC2FIX', 'IUVISCTR', 'IUVIS1FIX',
                               'IUVIS2FIX']
        data = {}
        reference_catalog = vstack([compute_sky_to_tel_in_table(obs.gaia_catalog_matched, attitude, obs.aperture,
                                                                use_tel_boresight=parameters['use_tel_boresight']) for
                                    obs in obs_set.observations], metadata_conflicts='silent')
        star_catalog = vstack([compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                           use_tel_boresight=parameters['use_tel_boresight'],
                                                           method=parameters['idl_tel_method']) for obs in
                               obs_set.observations], metadata_conflicts='silent')

        data['reference'] = {'x': np.array(reference_catalog['v2_tangent_arcsec']),
                             'y': np.array(reference_catalog['v3_tangent_arcsec'])}
        data['comparison_0'] = {'x': np.array(star_catalog['v2_tangent_arcsec']),
                                'y': np.array(star_catalog['v3_tangent_arcsec'])}
        plot_spatial_difference(data, siaf=parameters['siaf'], plot_aperture_names=plot_aperture_names)

    for i, index in enumerate(attitude_order_index):

        # with i increasing, the number of used observations increases
        obs_indices = attitude_order_index[0:i + 1]
        used_apertures = [name for name in aperture_names[obs_indices]]
        print('Attitude determination with {} apertures: {}'.format(len(obs_indices), used_apertures))
        iteration_number = 0
        # determine and correct for alignment parameters of individual apertures using current attitude

        skip_temporary_alignment_for_aligned_apertures = parameters['skip_temporary_alignment_for_aligned_apertures']
        # skipped_apertures = ['IUVIS1FIX', 'IUVIS2FIX', 'JWFC1FIX', 'JWFC2FIX']
        skipped_apertures = parameters['apertures_to_calibrate']

        if perform_alignment_update:
            print('+' * 20)
            for align_index in obs_indices:
                if verbose:
                    print('determine_attitude: Intermediate alignment of {}'.format(aperture_names[align_index]))
                align_obs = copy.deepcopy(obs_set.observations[align_index])
                if (skip_temporary_alignment_for_aligned_apertures is True) & (align_obs.aperture.AperName in skipped_apertures):
                    print('Skipping temporary alignment of {}. Using previously determined alignment and distortion.'.format(align_obs.aperture.AperName))
                    distortion_dict = align_obs.distortion_dict
                    for attribute in 'V2Ref V3Ref V3IdlYAngle'.split():
                        setattr(align_obs.aperture, '{}_corrected'.format(attribute), getattr(align_obs.aperture, '{}'.format(attribute)))
                    obs_set.observations[align_index].aperture = align_obs.aperture
                else:
                    print(align_obs.aperture.AperName)
                    # get V2/V3 tangent plane coordinates of Gaia stars
                    align_obs.gaia_catalog_matched = compute_sky_to_tel_in_table(align_obs.gaia_catalog_matched, attitude,
                        align_obs.aperture, use_tel_boresight=use_tel_boresight)

                    # here, attitude-def and alignment-ref aperture are set the same
                    alignment_reference_obs = copy.deepcopy(attitude_defining_obs)
                    alignment_reference_observation_index = attitude_defining_aperture_index
                    if verbose:
                        print('Attitude defining aperture {}; Alignment reference aperture {}; current '
                              'aperture '
                              '{}'.format(attitude_defining_obs.aperture.AperName,
                            alignment_reference_obs.aperture.AperName, align_obs.aperture.AperName))

                    plot_residuals = parameters['plot_residuals']
                    plot_dir = parameters['plot_dir']
                    # align aperture
                    determine_aperture_error(align_obs, alignment_reference_obs, align_index,
                                             alignment_reference_observation_index, plot_residuals=plot_residuals,
                                             plot_dir=plot_dir, verbose=verbose,
                                             idl_tel_method=parameters['idl_tel_method'],
                                             use_tel_boresight=parameters['use_tel_boresight'],
                                             reference_point_setting=parameters['reference_point_setting'],
                                             fractional_threshold_for_iterations=parameters['fractional_threshold_for_iterations'],
                                             parameters=parameters)


                    # transfer parameters of aligned aperture to observation set used for attitude
                    # determination
                    # no correction applied for reference aperture
                    if verbose:
                        print('-'*20)
                    obs_set.observations[align_index].aperture = align_obs.aperture
                    if align_index != attitude_defining_aperture_index:
                        if verbose:
                            print('Updating {} with temporary alignment parameters.'.format(
                            align_obs.aperture.AperName))

                        if (align_obs.aperture.observatory ==  'HST') & ('FGS' in align_obs.aperture.AperName):
                            alignment_attributes = alignment_definition_attributes['hst_fgs']
                        else:
                            alignment_attributes = alignment_definition_attributes['default']
                        for attribute in alignment_attributes:
                            corrected = getattr(align_obs.aperture, '{}_corrected'.format(attribute))
                            setattr(obs_set.observations[align_index].aperture, '{}'.format(attribute), corrected)
                    else:
                        if verbose:
                            print('No update of {} temporary alignment parameters (att-def and align-ref aperture).'.format(align_obs.aperture.AperName))


                # attach current distortion solution
                obs_set.observations[align_index].distortion_dict = align_obs.distortion_dict

        else:
            alignment_reference_obs = copy.deepcopy(attitude_defining_obs)

        print('+'*20)
        print('Determine refined attitude estimate:')
        # determine the current attitude
        for j in np.arange(maximum_number_of_iterations):
            if verbose:
                print('Iteration {}'.format(iteration_number))
            if iteration_number > 0:
                ra_attitude_deg += correction_ra_attitude_deg
                dec_attitude_deg += correction_dec_attitude_deg
                pa_attitude_deg += correction_V3IdlYAngle_deg

            attitude = pysiaf.rotations.attitude(V2Ref, V3Ref, ra_attitude_deg, dec_attitude_deg, pa_attitude_deg)

            reference_catalog = vstack([compute_sky_to_tel_in_table(obs.gaia_catalog_matched, attitude, obs.aperture,
                                                                    use_tel_boresight=parameters['use_tel_boresight'])
                                        for obs in obs_set.observations[obs_indices]], metadata_conflicts='silent')
            if perform_distortion_correction:
                star_catalog = vstack([compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                                   distortion_correction=obs.distortion_dict,
                                                                   use_tel_boresight=parameters['use_tel_boresight'],
                                                                   method=parameters['idl_tel_method']) for obs in
                                       obs_set.observations[obs_indices]], metadata_conflicts='silent')
            else:
                star_catalog = vstack([compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                                   use_tel_boresight=parameters['use_tel_boresight'],
                                                                   method=parameters['idl_tel_method']) for obs in
                                       obs_set.observations[obs_indices]], metadata_conflicts='silent')

            mp = pystortion.distortion.prepare_multi_epoch_astrometry(star_catalog, reference_catalog,
                                                                      fieldname_dict=fieldname_dict)


            # make sure to use consistent distortion reference point
            # reference_point = obs.distortion_dict['reference_point']
            if parameters['use_v1_pointing']:
                # here the reference point has to be on the V1 axis
                reference_point = np.array([[0., 0.], [0., 0.]])
            lazAC, index_masked_stars = pystortion.distortion.fit_distortion_general(mp, k,
                                                                                     eliminate_omc_outliers_iteratively=
                                                                                     parameters[
                                                                                         'eliminate_omc_outliers_iteratively'],
                                                                                     outlier_rejection_level_sigma=
                                                                                     parameters[
                                                                                         'outlier_rejection_level_sigma'],
                                                                                     reference_frame_number=parameters[
                                                                                         'reference_frame_number'],
                                                                                     evaluation_frame_number=parameters[
                                                                                         'evaluation_frame_number'],
                                                                                     reference_point=reference_point,
                                                                                     verbose=False)
            # verbose=parameters['verbose'])

            lazAC.human_readable_solution_parameters = pystortion.distortion.compute_rot_scale_skew(lazAC,
                i=parameters['evaluation_frame_number'])

            current_rotation_deg, current_shift_in_X, current_shift_in_Y, converged, sigma_current_rotation_deg, sigma_current_shift_in_X, sigma_current_shift_in_Y = determine_corrections(
                lazAC, parameters['fractional_threshold_for_iterations'], parameters['rotation_name'],
                verbose=parameters['verbose'])

            if (len(star_catalog) > 200) & (converged) & (0):
                # difference between applying distortion and not
                data = {}
                star_catalog = vstack([compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                                   distortion_correction=obs.distortion_dict,
                                                                   use_tel_boresight=parameters['use_tel_boresight'],
                                                                   method=parameters['idl_tel_method']) for obs in
                                       obs_set.observations[obs_indices]], metadata_conflicts='silent')
                data['reference'] = {'x': np.array(star_catalog['v2_tangent_arcsec']),
                                     'y': np.array(star_catalog['v3_tangent_arcsec'])}

                star_catalog = vstack([compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                                   distortion_correction=obs.distortion_dict,
                                                                   use_tel_boresight=parameters['use_tel_boresight'],
                                                                   # method=parameters[
                                                                   # 'idl_tel_method'],
                                                                   method='spherical_transformation', ) for obs in
                                       obs_set.observations[obs_indices]], metadata_conflicts='silent')

                data['comparison_0'] = {'x': np.array(star_catalog['v2_tangent_arcsec']),
                                        'y': np.array(star_catalog['v3_tangent_arcsec'])}

                plot_aperture_names = ['FGS1', 'FGS2', 'FGS3', 'JWFCFIX', 'JWFC1FIX', 'JWFC2FIX', 'IUVISCTR',
                                       'IUVIS1FIX', 'IUVIS2FIX']
                plot_spatial_difference(data, siaf=parameters['siaf'], plot_aperture_names=plot_aperture_names)

                1 / 0

            if converged:
                print('Attitude correction: Iterations converged after {} steps'.format(iteration_number))
                break

            correction_V3IdlYAngle_deg = attenuation_factor * current_rotation_deg

            ra_attitude_deg_intermediate, dec_attitude_deg_intermediate = pysiaf.rotations.pointing(attitude,
                V2Ref + attenuation_factor * current_shift_in_X, V3Ref + attenuation_factor * current_shift_in_Y)

            correction_ra_attitude_deg = attenuation_factor * (ra_attitude_deg_intermediate - ra_attitude_deg)
            correction_dec_attitude_deg = attenuation_factor * (dec_attitude_deg_intermediate - dec_attitude_deg)

            if verbose:
                print('correction_V3IdlYAngle_deg = {}'.format(correction_V3IdlYAngle_deg))
                print('correction_ra_attitude_deg = {}'.format(correction_ra_attitude_deg))
                print('correction_dec_attitude_deg = {}'.format(correction_dec_attitude_deg))

            iteration_number += 1

        if iteration_number == (maximum_number_of_iterations - 1):
            raise RuntimeError('Iterative attitude correction did not converge')
        else:
            if verbose | (i == len(attitude_order_index) - 1):
                print('=' * 50)
                print('Attitude correction: RESULTS')
                print('Attitude defining aperture {}; Alignment reference aperture {}'.format(
                    attitude_defining_obs.aperture.AperName, alignment_reference_obs.aperture.AperName))
                print('Attitude correction: Number of apertures {}; Number of stars {}'.format(len(used_apertures),
                    len(star_catalog)))
                RA_star_corr = (ra_attitude_deg - initial_ra_attitude_deg) * u.deg.to(u.arcsec) * u.arcsec * np.cos(
                    np.deg2rad(initial_dec_attitude_deg))
                Dec_corr = (dec_attitude_deg - initial_dec_attitude_deg) * u.deg.to(u.arcsec) * u.arcsec
                PA_corr = (pa_attitude_deg - initial_pa_attitude_deg) * u.deg.to(u.arcsec) * u.arcsec
                sigma_RA_star_corr = sigma_current_shift_in_X * u.arcsec
                sigma_Dec_corr = sigma_current_shift_in_Y * u.arcsec
                sigma_PA_corr = sigma_current_rotation_deg * u.deg.to(u.arcsec) * u.arcsec
                print('Attitude correction: {:.3f} in RA*cos(Dec), {:.3f} in Dec, {:.3f} in '
                      'PA_V3'.format(RA_star_corr, Dec_corr, PA_corr))
                print('Uncertainties: {:.3f} {:.3f} {:.3f}'.format(sigma_RA_star_corr, sigma_Dec_corr, sigma_PA_corr))
                print('{}, {}, {}, {:d}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
                    attitude_defining_obs.fpa_data.meta['EPOCH'], attitude_defining_aperture.InstrName,
                    attitude_defining_aperture.AperName, len(star_catalog), RA_star_corr.value,
                    sigma_RA_star_corr.value, Dec_corr.value, sigma_Dec_corr.value, PA_corr.value, sigma_PA_corr.value))
                print('=' * 50)
            # update attitude
            attitude = pysiaf.rotations.attitude(V2Ref, V3Ref, ra_attitude_deg, dec_attitude_deg, pa_attitude_deg)

    if parameters['show_final_fit']:
        # plot final residuals

        involved_apertures = [obs.aperture for obs in obs_set.observations[obs_indices]]
        res_titles = ['rms={:2.4f}'.format(r) for r in lazAC.rms[parameters['evaluation_frame_number']]]
        lazAC.plotResiduals(parameters['evaluation_frame_number'], parameters['plot_dir'], parameters['name_seed'],
                            title=res_titles, plot_apertures=involved_apertures)
        lazAC.display_results(print_rms_only=True)

    attitude_dict = {}
    attitude_dict['attitude'] = attitude
    attitude_dict['ra_deg'] = ra_attitude_deg
    attitude_dict['pa_deg'] = pa_attitude_deg
    attitude_dict['dec_deg'] = dec_attitude_deg
    attitude_dict['ra_star_arcsec_correction'] = RA_star_corr
    attitude_dict['dec_arcsec_correction'] = Dec_corr
    attitude_dict['pa_arcsec_correction'] = PA_corr
    attitude_dict['sigma_ra_star_arcsec_correction'] = sigma_RA_star_corr
    attitude_dict['sigma_dec_arcsec_correction'] = sigma_Dec_corr
    attitude_dict['sigma_pa_arcsec_correction'] = sigma_PA_corr
    attitude_dict['apertures'] = aperture_names
    attitude_dict['n_stars_total'] = len(star_catalog)
    attitude_dict['fit_residual_rms'] = lazAC.rms[parameters['evaluation_frame_number']]
    attitude_dict['aligned_obs_set'] = obs_set

    # show offset between measured positions and reference positions using final attitude
    if 0:
        plot_aperture_names = ['FGS1', 'FGS2', 'FGS3', 'JWFCFIX', 'JWFC1FIX', 'JWFC2FIX', 'IUVISCTR', 'IUVIS1FIX',
                               'IUVIS2FIX']
        data = {}
        reference_catalog = vstack([compute_sky_to_tel_in_table(obs.gaia_catalog_matched, attitude, obs.aperture,
                                                                use_tel_boresight=parameters['use_tel_boresight']) for
                                    obs in obs_set.observations], metadata_conflicts='silent')
        star_catalog = vstack([compute_idl_to_tel_in_table(obs.star_catalog_matched, obs.aperture,
                                                           use_tel_boresight=parameters['use_tel_boresight'],
                                                           method=parameters['idl_tel_method']) for obs in
                               obs_set.observations], metadata_conflicts='silent')

        data['reference'] = {'x': np.array(reference_catalog['v2_tangent_arcsec']),
                             'y': np.array(reference_catalog['v3_tangent_arcsec'])}
        data['comparison_0'] = {'x': np.array(star_catalog['v2_tangent_arcsec']),
                                'y': np.array(star_catalog['v3_tangent_arcsec'])}
        plot_spatial_difference(data, siaf=parameters['siaf'], plot_aperture_names=plot_aperture_names)

    return attitude_dict


def determine_corrections(lazAC, fractional_threshold_for_iterations, rotation_name, verbose=False):
    """Helper function for focal plane alignment. Based on the result of a 2D polynomial fit,
    convergence is
    established based on the amplitude of the remaining offsets and rotation compared to their
    uncertainties

    Parameters
    ----------
    lazAC
    fractional_threshold_for_iterations
    rotation_name
    verbose

    Returns
    -------

    """
    converged = False

    values = lazAC.human_readable_solution_parameters['values']
    names = lazAC.human_readable_solution_parameters['names'].tolist()

    current_rotation_deg = values[names.index(rotation_name)][0]
    sigma_current_rotation_deg = values[names.index(rotation_name)][1]

    current_shift_in_X = values[names.index('Shift in X')][0]
    sigma_current_shift_in_X = values[names.index('Shift in X')][1]

    current_shift_in_Y = values[names.index('Shift in Y')][0]
    sigma_current_shift_in_Y = values[names.index('Shift in Y')][1]

    if verbose:
        print('               shift in X: {:3.6f} \t shift in Y: {:3.6f} \t {} {:3.6f} deg'.format(current_shift_in_X,
            current_shift_in_Y, rotation_name, current_rotation_deg))
        print('uncertainties: shift in X: {:3.6f} \t shift in Y: {:3.6f} \t {} {:3.6f} deg'.format(
            sigma_current_shift_in_X, sigma_current_shift_in_Y, rotation_name, sigma_current_rotation_deg))

    if (np.abs(current_rotation_deg) < fractional_threshold_for_iterations * sigma_current_rotation_deg) & (
        np.abs(current_shift_in_X) < fractional_threshold_for_iterations * sigma_current_shift_in_X) & (
        np.abs(current_shift_in_Y) < fractional_threshold_for_iterations * sigma_current_shift_in_Y):
        converged = True

    return current_rotation_deg, current_shift_in_X, current_shift_in_Y, converged, sigma_current_rotation_deg, sigma_current_shift_in_X, sigma_current_shift_in_Y


def determine_focal_plane_alignment(obs_collection, parameters):
    """

    Parameters
    ----------
    obs_collection
    parameters

    Returns
    -------
    obs_collection:
        updated observation collection

    """
    simultaneous_attitude_alignment_determination = True
    exclude_fgs_from_attitude_determination = False

    alignment_reference_aperture_name = parameters['alignment_reference_aperture_name']
    attitude_defining_aperture_name = parameters['attitude_defining_aperture_name']
    restrict_to_sets_that_include_aperture = parameters['restrict_to_sets_that_include_aperture']
    calibration_alignment_reference_aperture_name = parameters['calibration_alignment_reference_aperture_name']
    calibration_attitude_defining_aperture_name = parameters['calibration_attitude_defining_aperture_name']
    program_id = parameters['program_id']
    correct_dva = parameters['correct_dva']
    k = parameters['k']
    k_attitude_determination = parameters['k_attitude_determination']
    result_dir = parameters['result_dir']
    original_siaf = parameters['original_siaf']
    random_realisations = parameters['random_realisations']
    overwrite_alignment_results_pickle = parameters['overwrite']
    plot_dir = parameters['plot_dir']
    make_summary_figures = parameters['make_summary_figures']
    overwrite_attitude_pickle = parameters['overwrite_attitude_pickle']
    rotation_name = parameters['rotation_name']
    write_calibration_result_file = parameters['write_calibration_result_file']
    visit_groups = parameters['visit_groups']
    apply_fpa_calibration = parameters['apply_fpa_calibration']
    apertures_to_calibrate = parameters['apertures_to_calibrate']
    calibration_field_selection = parameters['calibration_field_selection']

    if attitude_defining_aperture_name is None:
        raise NotImplementedError

    siaf = copy.deepcopy(original_siaf)

    attitude_determination_parameters = parameters['attitude_determination_parameters']
    attitude_determination_parameters['siaf'] = siaf  # used for plotting only


    # names of apertures that have to be present in any given observation set
    required_aperture_names = [alignment_reference_aperture_name, attitude_defining_aperture_name]
    if restrict_to_sets_that_include_aperture is not None:
        required_aperture_names.append(restrict_to_sets_that_include_aperture)

    applied_calibration_file = None
    if apply_fpa_calibration & (calibration_alignment_reference_aperture_name is not None):
        temp_seed = '{}_{}_alignref_{}_attdef_{}_k{}_k{}_HSTFGS{}'.format(program_id, parameters['idl_tel_method'],
            calibration_alignment_reference_aperture_name, calibration_attitude_defining_aperture_name, k, k_attitude_determination,
            str(parameters['use_hst_fgs_fiducial_as_reference_point'][0]))
        applied_calibration_file = '{}_calibration_complete.csv'.format(temp_seed)

        calibrated_data = Table.read(os.path.join(result_dir, applied_calibration_file), format='ascii.basic', delimiter=',')
        if obs_collection.observatory == 'JWST':
            # this is for multi-epoch trending of focal plane alignment data.
            # TODO fix the setting of this parameter
            calibrated_data['EPOCHNUMBER'] = 0
        else:
            calibrated_data['EPOCHNUMBER'] = [str(s)[0] for s in calibrated_data['VISIT']]
        calibrated_obs_collection = pickle.load(open(os.path.join(result_dir, 'alignment_results_{}.pkl'.format(temp_seed)), "rb"))



    # create seed for file and figure naming
    figure_filename_tag = '{}_{}'.format(program_id, parameters['idl_tel_method'])
    figure_filename_tag += '_alignref_{}_attdef_{}'.format(alignment_reference_aperture_name,
                                                           attitude_defining_aperture_name)
    figure_filename_tag += '_k{}_k{}_HSTFGS{}'.format(k, k_attitude_determination, str(parameters['use_hst_fgs_fiducial_as_reference_point'][0]))

    if applied_calibration_file is not None:
        figure_filename_tag += '_CALIBRATION_APPLIED'

    obs_collection_original = copy.deepcopy(obs_collection)
    figure_filename_tag_original = copy.deepcopy(figure_filename_tag)

    tvs_results = {}

    # if apply_fpa_calibration & (applied_calibration_file is not None):
    #     calibrated_data = Table.read(os.path.join(result_dir, applied_calibration_file), format='ascii.basic', delimiter=',')
    #     calibrated_data['EPOCHNUMBER'] = [str(s)[0] for s in calibrated_data['VISIT']]

    # DETERMINE THE ALIGNMENT PARAMETERS
    for random_realisation in random_realisations:
        obs_collection = copy.deepcopy(obs_collection_original)

        obs_collection.T['EPOCHNUMBER'] = [s.split('_')[1][0] for s in obs_collection.T['PROGRAM_VISIT']]

        # update aperture alignment parameters based on previous alignment
        if apply_fpa_calibration:
            obs_collection = apply_focal_plane_calibration(obs_collection, apertures_to_calibrate, calibrated_data,
                                                           verbose=True, field_selection=calibration_field_selection,
                                                           calibrated_obs_collection=calibrated_obs_collection)

        # create a bootstrapped dataset by removing one star at a time
        if random_realisation != 0:
            obs_collection = remove_star_random_realisation(obs_collection, random_realisation)
            figure_filename_tag = copy.deepcopy(figure_filename_tag_original) + '_realisation{}'.format(random_realisation)

        alignment_results_pickle_file = os.path.join(result_dir, 'alignment_results_{}.pkl'.format(figure_filename_tag))
        if (not os.path.isfile(alignment_results_pickle_file)) | overwrite_alignment_results_pickle:
            # MAIN LOOP THAT DETERMINES THE ALIGNMENT PARAMETERS
            # analyse observations by group, which allows to assign the alignment reference attitude

            for exclusive_aperture_name in required_aperture_names:
                # select only group_ids in which the alignment_reference_aperture was used,
                # especially relevant for FGS
                selected_group_ids = np.unique(np.array(
                    obs_collection.T[np.where((obs_collection.T['AperName'] == exclusive_aperture_name))]['group_id']))
                selected_observations = np.where(np.in1d(obs_collection.T['group_id'], selected_group_ids))[0]
                discarded_observations = np.setdiff1d(np.arange(len(obs_collection.T)), selected_observations)

                # remove observations that are not used in case FGS is alignment reference
                obs_collection.delete_observations(discarded_observations)

            for group_id in selected_group_ids:
                obs_indexes = np.where((obs_collection.T['group_id'] == group_id))[0]

                if 0:
                    alignment_reference_observation_index = \
                    np.where((obs_collection.T['group_id'] == group_id) & (obs_collection.T['alignment_reference'] == 1))[0]
                    if (np.ndim(alignment_reference_observation_index) == 1):
                        # catch case where alignment_reference_observation_index is an array of
                        # several elements
                        # Here we select the first one (arbitrarily).
                        alignment_reference_observation_index = alignment_reference_observation_index[0]

                    alignment_reference_obs = obs_collection.observations[alignment_reference_observation_index]

                    # attitude_defining_aperture = siaf.apertures[attitude_defining_aperture_name]

                    # For cameras, there are usually two exposures that could be used to determine
                    # the attitude.
                    # Here we select the first one (arbitrarily).

                    attitude_defining_observation_index = np.where((obs_collection.T['group_id'] == group_id) & (
                        obs_collection.T['AperName'] == attitude_defining_aperture_name))[0]
                    if (np.ndim(attitude_defining_observation_index) == 1):
                        attitude_defining_observation_index = attitude_defining_observation_index[0]
                    attitude_defining_obs = obs_collection.observations[attitude_defining_observation_index]

                    # ensure that the reference observation is being processed first
                    obs_indexes_list = obs_indexes.tolist()
                    obs_indexes_list.remove(alignment_reference_observation_index)
                    obs_indexes_list.insert(0, alignment_reference_observation_index)
                    obs_indexes = np.array(obs_indexes_list)

                print('=' * 100)
                print('processing GROUP {}'.format(group_id))
                # determine attitude and alignment iteratively for camera apertures
                if simultaneous_attitude_alignment_determination:
                    obs_subset = copy.deepcopy(obs_collection)
                    obs_subset.select_observations(obs_indexes)
                    obs_subset.T.pprint()

                    attitude_groups = np.unique(obs_subset.T['attitude_group'])
                    for attitude_group in attitude_groups:
                        pl.close('all')
                        if attitude_group < 0:
                            continue
                        attitude_determination_parameters['name_seed'] = '{}_group_{}-{}'.format(figure_filename_tag,
                            group_id, attitude_group)
                        attitude_pickle_file = os.path.join(result_dir, 'corrected_attitude_{}.pkl'.format(
                            attitude_determination_parameters['name_seed']))

                        if exclude_fgs_from_attitude_determination:
                            set_index = np.where((obs_subset.T['attitude_group'] == attitude_group) & (
                            obs_subset.T['INSTRUME'] != 'SUPERFGS'))[0]
                        else:
                            set_index = np.where(obs_subset.T['attitude_group'] == attitude_group)[0]

                        if (not os.path.isfile(attitude_pickle_file)) | overwrite_attitude_pickle:
                            obs_set = copy.deepcopy(obs_subset)
                            obs_set.select_observations(set_index)
                            if attitude_determination_parameters['use_fgs_pseudo_aperture']:
                                # use pseudo aperture for FGS
                                for i, obs in enumerate(obs_set.observations):
                                    if 'FGS' in obs.aperture.AperName:
                                        obs.aperture = obs.aperture.pseudo_aperture
                                        obs_set.observations[i] = obs

                            corrected_attitude = determine_attitude(obs_set, attitude_determination_parameters)

                            if 1:
                                for j in range(obs_set.n_observations):
                                    original_aperture = obs_set.observations[j].aperture
                                    corrected_aperture = corrected_attitude['aligned_obs_set'].observations[j].aperture
                                    print('{:>15} differences:'.format(original_aperture.AperName), end=' ')
                                    for key in ['V3IdlYAngle', 'V2Ref', 'V3Ref']:
                                        print('{} is {:+2.4f}'.format(key, getattr(corrected_aperture, key) - getattr(
                                            original_aperture, key)), end=' ')
                                    print('')

                            pickle.dump((corrected_attitude), open(attitude_pickle_file, "wb"))
                        else:
                            corrected_attitude = pickle.load(open(attitude_pickle_file, "rb"))
                            print('Loaded results from pickled file {}'.format(attitude_pickle_file))

                        # 1/0
                        # transfer corrected attitude information from subset to obs_collection
                        # if attitude_group == 1:
                        #     1/0
                        for s_index in set_index:
                            obs_coll_index = obs_collection.T['INDEX'].tolist().index(obs_subset.T['INDEX'][s_index])
                            obs_collection.observations[obs_coll_index].corrected_attitude = corrected_attitude
                            # print('{} {} {} {}'.format(s_index, obs_coll_index, obs_collection.T['INSTRUME'][obs_coll_index], obs_collection.observations[obs_coll_index].corrected_attitude['apertures']))
                            # obs_collection.T['attitude_group'][obs_coll_index] = attitude_group

                # determine corrections to aperture parameters -> focal plane geometric calibration
                # plot_residuals = False
                plot_residuals = parameters['plot_residuals']
                # verbose = False
                verbose = parameters['verbose']
                attitude_groups = np.unique(obs_collection.T['attitude_group'][obs_indexes])
                for attitude_group in attitude_groups:
                    if attitude_group < 0:
                        1/0
                    set_indexes = np.where((obs_collection.T['attitude_group'] == attitude_group) &
                                           (obs_collection.T['group_id'] == group_id))[0]

                    # check that aperture names are unique in each set
                    if len(np.unique(obs_collection.T['AperName'][set_indexes])) != len(set_indexes):
                        raise RuntimeError

                    alignment_reference_observation_index = set_indexes[obs_collection.T['alignment_reference'][set_indexes] == 1][0]
                    alignment_reference_obs = obs_collection.observations[alignment_reference_observation_index]

                    # ensure that the reference observation is being processed first
                    set_indexes_list = set_indexes.tolist()
                    set_indexes_list.remove(alignment_reference_observation_index)
                    set_indexes_list.insert(0, alignment_reference_observation_index)
                    set_indexes = np.array(set_indexes_list)

                    for jj, set_index in enumerate(set_indexes):
                        pl.close('all')
                        obs = obs_collection.observations[set_index]

                        if obs.aperture.AperName in 'FGS1 FGS2 FGS3'.split():
                            attitude_index = np.where((obs_collection.T['attitude_id'][set_index] == obs_collection.T['attitude_id']) &
                                                      (obs_collection.T['INSTRUME'] != 'SUPERFGS'))[0][0]
                            alignment_reference_attitude = obs_collection.observations[attitude_index].corrected_attitude['attitude']
                        else:
                            alignment_reference_attitude = copy.deepcopy(obs.corrected_attitude['attitude'])

                        # compute V2,V3 coordinates of Gaia stars using current attitude
                        obs.gaia_catalog_matched = compute_sky_to_tel_in_table(obs.gaia_catalog_matched,
                            alignment_reference_attitude, obs.aperture)

                        print('+' * 20)
                        print('Attitude defining aperture {}; Alignment reference aperture {}; '
                              'current aperture {}'.format(attitude_defining_aperture_name,
                                                           alignment_reference_aperture_name,
                                                           obs.aperture.AperName))


                        determine_aperture_error(obs, alignment_reference_obs, set_index,
                                                 alignment_reference_observation_index,
                                                 alignment_reference_aperture=alignment_reference_aperture_name,
                                                 plot_residuals=plot_residuals, plot_dir=plot_dir, verbose=verbose, k=k,
                                                 idl_tel_method=attitude_determination_parameters['idl_tel_method'],
                                                 use_tel_boresight=attitude_determination_parameters['use_tel_boresight'],
                                                 reference_point_setting=attitude_determination_parameters['reference_point_setting'],
                                                 fractional_threshold_for_iterations=attitude_determination_parameters['fractional_threshold_for_iterations'],
                                                 parameters=attitude_determination_parameters)

                        if obs.aperture.AperName == 'FGS33':
                            obs.lazAC.plot_distortion_offsets()
                            xx = obs.star_catalog_matched['v2_spherical_arcsec']
                            yy = obs.star_catalog_matched['v3_spherical_arcsec']
                            xx_transformed, yy_transformed = obs.lazAC.apply_polynomial_transformation(1, xx, yy)
                            print(xx_transformed)
                            print(yy_transformed)
                            obs.gaia_catalog_matched['v2_spherical_arcsec', 'v3_spherical_arcsec'].pprint()


            # generate auxiliary columns in result table
            obs_collection = enhance_result_table(obs_collection)

            # dictionary to hold information necessary to produce analysis figures and metrics
            info_dict = {}
            info_dict['program_id'] = program_id
            info_dict['random_realisation'] = random_realisation
            info_dict['random_realisations'] = random_realisations
            info_dict['visit_groups'] = visit_groups
            info_dict['k'] = k
            info_dict['correct_dva'] = correct_dva
            info_dict['alignment_reference_aperture_name'] = alignment_reference_aperture_name
            info_dict['attitude_defining_aperture_name'] = attitude_defining_aperture_name
            info_dict['figure_filename_tag'] = figure_filename_tag
            info_dict['rotation_name'] = rotation_name
            info_dict['visit_groups_parameters'] = {}
            info_dict['visit_groups_parameters']['star_marker'] = ['.', 'o', 'x', '+']
            info_dict['visit_groups_parameters']['color'] = ['b', 'g', '0.7', 'k']

            obs_collection.info_dict = info_dict

            pickle.dump((obs_collection), open(alignment_results_pickle_file, "wb"))
            print('Wrote results to pickled file {}'.format(alignment_results_pickle_file))
        else:
            obs_collection = pickle.load(open(alignment_results_pickle_file, "rb"))
            print('Loaded results from pickled file {}'.format(alignment_results_pickle_file))

        # ======================================================================================================
        if write_calibration_result_file:
            username = os.getlogin()
            timestamp = Time.now()

            obs_collection.T['VISIT'] = [s.split('_')[1] for s in obs_collection.T['PROGRAM_VISIT']]
            obs_collection.T['align_ref_aperture'] = alignment_reference_aperture_name
            obs_collection.T['attitude_def_aperture'] = attitude_defining_aperture_name
            obs_collection.T['applied_calibration_file'] = applied_calibration_file
            comments = []
            comments.append('Focal plane alignment calibration reference file.')
            comments.append('')
            comments.append('Attitude defining aperture {}'.format(attitude_defining_aperture_name))
            comments.append('Alignment reference aperture {}'.format(alignment_reference_aperture_name))
            comments.append('Calibration file applied to alignment reference is {}'.format(applied_calibration_file))
            comments.append('')
            comments.append('Generated {} {}'.format(timestamp.isot, timestamp.scale))
            comments.append('{}'.format(username))
            comments.append('')

            obs_collection.T.meta['comments'] = comments
            keys_for_calibration_file = 'PROPOSID VISIT DATE-OBS TELESCOP INSTRUME CHIP AperName ' \
                                        'SIAFAPER alignment_reference align_ref_aperture ' \
                                        'attitude_def_aperture applied_calibration_file'.split()
            # for key in 'V3IdlYAngle V2Ref V3Ref'.split():
            for key in 'v2_position_arcsec v3_position_arcsec v3_angle_arcsec'.split():
                keys_for_calibration_file.append(key)
                # keys_for_calibration_file.append('{}_corrected'.format(key))
                # keys_for_calibration_file.append('sigma_{}_corrected'.format(key))
                keys_for_calibration_file.append('calibrated_{}'.format(key))
                keys_for_calibration_file.append('sigma_calibrated_{}'.format(key))

            # obs_collection.T[keys_for_calibration_file].write(sys.stdout,
            # format='ascii.fixed_width', delimiter=',', bookend=False)

            calibration_file = os.path.join(result_dir, '{}_calibration.csv'.format(figure_filename_tag))
            obs_collection.T[keys_for_calibration_file].write(calibration_file, format='ascii.fixed_width',
                                                              delimiter=',', bookend=False, overwrite=True)
            calibration_file = os.path.join(result_dir, '{}_calibration_complete.csv'.format(figure_filename_tag))
            obs_collection.T.write(calibration_file, format='ascii.fixed_width', delimiter=',', bookend=False,
                                   overwrite=True)

        # present results
        if make_summary_figures:
            tvs_results = evaluate(obs_collection, make_summary_figures=make_summary_figures, save_plot=save_plot,
                                   plot_dir=plot_dir, tvs_results=tvs_results)

    # print estimates of tvs parameter uncertainties
    if len(random_realisations) != 1:
        tvs_uncertainties(tvs_results, result_dir=result_dir)

    return obs_collection




def remove_star_random_realisation(obs_collection, random_realisation, verbose=True):
    for jj in range(len(obs_collection.observations)):
        if verbose:
            print('Realisation {}: removing star in row index {} of {} rows from aperture {}'.format(random_realisation,
                random_realisation - 1, len(obs_collection.observations[jj].star_catalog_matched),
                obs_collection.observations[jj].aperture.AperName))
        obs_collection.observations[jj].star_catalog_matched.remove_row(random_realisation - 1)
        obs_collection.observations[jj].gaia_catalog_matched.remove_row(random_realisation - 1)
    return obs_collection


def enhance_result_table(obs_collection):
    """Generate auxiliary columns in result table.

    Parameters
    ----------
    obs_collection

    Returns
    -------

    """
    table = copy.deepcopy(obs_collection.T)
    n_obs = obs_collection.n_observations
    observations = obs_collection.observations

    attribute_mapping = {'V3IdlYAngle': 'rotation_deg',
                         'V2Ref': 'shift_in_X',
                         'V3Ref': 'shift_in_Y',}

    zero_array = np.zeros(len(table))

    # # homogenize offset and clocking parameter naming across HST cameras and guiders
    # table['hst_fgs'] = (table['TELESCOP']=='HST') & (np.array([True if 'FGS' in a else False for a in table['APERTURE']]))
    # table['align_params'] = ['hst_fgs' if a else 'default' for a in table['hst_fgs']]

    for key, attribute in alignment_parameter_mapping['default'].items():
        table[key] = [getattr(observations[j].aperture, alignment_parameter_mapping[table['align_params'][j]][key]) for j in range(n_obs)]
        table['corrected_{}'.format(key)] = [getattr(observations[j].aperture, '{}_corrected'.format(alignment_parameter_mapping[table['align_params'][j]][key])) for j in range(n_obs)]
        table['sigma_corrected_{}'.format(key)] = [getattr(observations[j], 'sigma_current_{}'.format(attribute_mapping[attribute])) for j in range(n_obs)]

    columns_to_show = ['APERTURE']
    for flavour in ['', 'corrected_', 'sigma_corrected_']:
        for key, attribute in alignment_parameter_mapping['default'].items():
            # columns_to_show.append('{}{}'.format(flavour, key))
            key_arcsec = '{}{}_arcsec'.format(flavour, key)
            columns_to_show.append(key_arcsec)
            # convert to arcseconds
            table[key_arcsec] = table['{}{}'.format(flavour, key)] * alignment_parameter_mapping['unit'][key].to(u.arcsec)

    flavour = 'corrected_'
    for key, attribute in alignment_parameter_mapping['default'].items():
        key_arcsec = '{}_arcsec'.format(key)
        flavour_key_arcsec = '{}{}_arcsec'.format(flavour, key)
        sigma_flavour_key_arcsec = 'sigma_{}{}_arcsec'.format(flavour, key)

        # See ISR: this is (p_\mathrm{ref, corrected} - p_\mathrm{ref, SIAF})
        # this is the correction offset introduced by adjusting the alignment parameter to the
        # alignment attitude.
        table['delta_{}'.format(flavour_key_arcsec)] = table[flavour_key_arcsec] - table[key_arcsec]
        table['sigma_delta_{}'.format(flavour_key_arcsec)] = table[sigma_flavour_key_arcsec]

        # initialize columns for next step, which is reporting difference relative to alignment
        # reference
        table['calibrated_{}'.format(key_arcsec)] = zero_array
        table['sigma_calibrated_{}'.format(key_arcsec)] = zero_array


    # compute differences relative to alignment reference observation (in every attitude group)
    for attitude_id in np.unique(table['attitude_id']):
        obs_index = np.where(table['attitude_id'] == attitude_id)[0]
        alignment_reference_observation_index = np.where((table['attitude_id'] == attitude_id) & (table['alignment_reference'] == 1))[0]
        if len(alignment_reference_observation_index) != 1:
            raise RuntimeError('Too many alignment references in attitude group')
        ref_index = alignment_reference_observation_index[0]

        for key, attribute in alignment_parameter_mapping['default'].items():
            key_arcsec = '{}_arcsec'.format(key)
            if key == 'v3_angle':
                # calibrated clocking angle is the same as corrected, because effect of rotation
                # is not constant across the field (as opposed to an offset)
                # for reference aperture, clocking angle is forced to remain at SIAF/amu.rep value
                # table['calibrated_{}'.format(key_arcsec)][obs_index] = table['corrected_{}'.format(key_arcsec)][obs_index]
                table['calibrated_{}'.format(key_arcsec)][obs_index] = [table['corrected_{}'.format(key_arcsec)][j] if j!=ref_index else table['{}'.format(key_arcsec)][j] for j in obs_index]
            else:
                if 1:
                    table['calibrated_{}'.format(key_arcsec)][obs_index] = [table['corrected_{}'.format(key_arcsec)][j]
                                                                        - (table['corrected_{}'.format(key_arcsec)][ref_index] - table[key_arcsec][ref_index]) for j in obs_index]
                else:
                    for obs_i in obs_index:
                        if table['INSTRUME'][obs_i] == 'SUPERFGS':
                            1/0
                            table['calibrated_{}'.format(key_arcsec)][obs_i] = table['corrected_{}'.format(key_arcsec)][obs_i]
                                # - (
                                #     table['corrected_{}'.format(key_arcsec)][ref_index] -
                                #     table[key_arcsec][ref_index]) for j in [obs_i]]

                        else:
                            table['calibrated_{}'.format(key_arcsec)][obs_i] = table['corrected_{}'.format(key_arcsec)][obs_i] - (
                                table['corrected_{}'.format(key_arcsec)][ref_index] - table[key_arcsec][ref_index])

            # REVISIT THIS, seems to overestimate uncertainty for reference aperture
            # uncertainty
            table['sigma_calibrated_{}'.format(key_arcsec)][obs_index] = [np.sqrt((table['sigma_corrected_{}'.format(key_arcsec)][ref_index]) ** 2
                                                                               + (table['sigma_corrected_{}'.format(key_arcsec)][j]) ** 2) for j in obs_index]

    # Offset of calibrated from current/SIAF value (zero for reference aperture)
    # See ISR: this is p_{i,\mathrm{calibrated}} - p_{i,\mathrm{SIAF}}
    for key, attribute in alignment_parameter_mapping['default'].items():
        key_arcsec = '{}_arcsec'.format(key)
        table['delta_calibrated_{}'.format(key_arcsec)] = table['calibrated_{}'.format(key_arcsec)] - table[key_arcsec]
        table['sigma_delta_calibrated_{}'.format(key_arcsec)] = table['sigma_calibrated_{}'.format(key_arcsec)]

    # generate table entries with instrument specific names and deg/arcsec units
    flavour = 'calibrated_'
    for key, attribute in alignment_parameter_mapping['default'].items():
        new_column_name = '{}{}'.format(flavour, attribute)
        if new_column_name in table.colnames:
            raise ValueError
        table[new_column_name] = zero_array
        new_column_name = '{}{}'.format(flavour, alignment_parameter_mapping['hst_fgs'][key])
        if new_column_name in table.colnames:
            raise ValueError
        table[new_column_name] = zero_array

    # fill in values and convert to degree when necessary
    flavour = 'calibrated_'
    for row in range(len(table)):
        for key, attribute in alignment_parameter_mapping[table['align_params'][row]].items():
            column_name = '{}{}_arcsec'.format(flavour, key)
            new_column_name = '{}{}'.format(flavour, attribute)
            if key == 'v3_angle':
                factor = u.arcsec.to(u.deg)
            else:
                factor = 1.
            table[new_column_name][row] = table[column_name][row] * factor


    # Retrieve scales from distortion fit
    lazac_mapping = {'scale_in_x': 'Scale in X',
                     'scale_in_y': 'Scale in Y',
                     'rotation_in_x': 'Rotation in X',
                     'rotation_in_y': 'Rotation in Y',
                     'on_axis_skew': 'On-axis Skew',
                     'off_axis_skew': 'Off-axis Skew',}
    for key in lazac_mapping:
        table[key] = [getattr(observations[j], 'lazAC').human_readable_solution_parameters['values']
                      [observations[j].lazAC.human_readable_solution_parameters['names'].tolist().index(lazac_mapping[key])][0]
                      for j in range(n_obs)]
        table['sigma_{}'.format(key)] = [getattr(observations[j], 'lazAC').human_readable_solution_parameters['values']
                      [observations[j].lazAC.human_readable_solution_parameters['names'].tolist().index(lazac_mapping[key])][1]
                                         for j in range(n_obs)]

    for key in ['number_of_used_stars_for_aperture_correction', 'number_of_measured_stars',
                'number_of_gaia_stars', 'number_of_matched_stars']:
        table[key] = [getattr(observations[j], key) for j in range(n_obs)]

    obs_collection.T = copy.deepcopy(table)

    return obs_collection


def evaluate(obs_collection, parameters, make_summary_figures=True, save_plot=True, plot_dir='', comparison_tvs_data=None, tvs_results={}, print_detailed_results=False):
    """Evaluate alignment solution and generate figures and tables.

    Parameters
    ----------
    obs_collection
    make_summary_figures
    save_plot
    plot_dir
    comparison_tvs_matrix
    tvs_results

    Returns
    -------

    """

    if (comparison_tvs_data is None) and (obs_collection.observatory=='HST'):
        comparison_tvs_data = pysiaf.read.read_hst_fgs_amudotrep()

    random_realisation = obs_collection.info_dict['random_realisation']
    random_realisations = obs_collection.info_dict['random_realisations']
    alignment_reference_aperture_name = obs_collection.info_dict['alignment_reference_aperture_name']
    attitude_defining_aperture_name = obs_collection.info_dict['attitude_defining_aperture_name']
    figure_filename_tag = obs_collection.info_dict['figure_filename_tag']
    visit_groups = obs_collection.info_dict['visit_groups']
    siaf = obs_collection.info_dict['siaf']
    rotation_name = obs_collection.info_dict['rotation_name']

    tvs_results['visit_groups'] = visit_groups
    observatory = obs_collection.observatory

    magnification_factor = parameters['magnification_factor']

    for group_id in np.unique(obs_collection.T['group_id']):
        obs_index = np.where((obs_collection.T['group_id'] == group_id))[0]
        reference_observation_index = np.where((obs_collection.T['group_id'] == group_id) & (
        obs_collection.T['alignment_reference'] == 1))[0]
        if (np.ndim(reference_observation_index) == 1):
            # catch case where alignment_reference_observation_index is an array of several elements
            reference_observation_index = reference_observation_index[0]

        reference_obs = obs_collection.observations[reference_observation_index]

        alignment_reference_aperture = reference_obs.aperture
        if random_realisation == 0:
            if print_detailed_results:
                print('=' * 100)
                print('Showing results for GROUP {}'.format(group_id))
                print('Focal plane alignment parameters relative to {}'.format(
                    alignment_reference_aperture_name))
            for j, obs in enumerate(obs_collection.observations[obs_index]):
                # if obs.aperture.AperName != alignment_reference_aperture_name:
                # if j != alignment_reference_aperture_index:
                if print_detailed_results:
                    print('Parameters are reported as {1}-{0}'.format(obs.aperture.AperName,
                                                                      alignment_reference_aperture_name))
                    for attribute in ['V3IdlYAngle', 'V2Ref', 'V3Ref']:
                        current_difference = getattr(alignment_reference_aperture,
                                                     '{}'.format(attribute)) - getattr(obs.aperture,
                                                                                       '{}'.format(
                                                                                           attribute))
                        corrected_difference = getattr(alignment_reference_aperture,
                                                       '{}_corrected'.format(attribute)) - getattr(
                            obs.aperture, '{}_corrected'.format(attribute))
                        correction_amplitude = corrected_difference - current_difference
                        print(
                            'current SIAF difference in {}: {:3.3f} \t corrected difference {'
                            ':3.3f} \t correction amplitude {:3.3f}'.format(
                                attribute, current_difference, corrected_difference,
                                correction_amplitude))

    # visualize effect of attitude error correction (it is almost invisible)
    if 0:
        attitude_parameters = [(RA_APER_deg, DEC_APER_deg, PA_V3_deg), (
            reference_obs.fpa_data.meta['RA_APER'], reference_obs.fpa_data.meta['DEC_APER'],
            reference_obs.fpa_data.meta['PA_V3'])]
        gaia_ref_cats = []
        for jj in range(len(attitude_parameters)):
            RA_APER_deg, DEC_APER_deg, PA_V3_deg = attitude_parameters[jj]
            # the aperture used as reference, thus used to correct for attitude errors
            alignment_reference_attitude = pysiaf.utils.rotations.attitude(
                attitude_defining_aperture.V2Ref,
                attitude_defining_aperture.V3Ref,
                RA_APER_deg, DEC_APER_deg, PA_V3_deg)
            aperture = obs_collection.observations[reference_observation_index][0].aperture
            gaia_ref_cats.append(
                fpalign.alignment.compute_sky_to_tel_in_table(gaia_reference_catalog[0::10],
                                                              alignment_reference_attitude,
                                                              aperture))

    if make_summary_figures:
        # make a visualisation plot in V2/V3
        # jj = 0

        offset_specifiers = parameters['offset_specifiers']
        figure_filename_tag_orig = copy.deepcopy(figure_filename_tag)

        for offset_specifier in offset_specifiers:
            figure_filename_tag = copy.deepcopy(figure_filename_tag_orig)
            fig = pl.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
            pl.clf()

            unique_aperture_names_all = np.unique(obs_collection.T['AperName'])

            for i in range(obs_collection.n_observations):
                visit_group_index = [j for j in range(len(visit_groups)) if obs_collection.T['group_id'][i] in visit_groups[j]][0]
                obs = obs_collection.observations[i]
                use_pseudo_fgs = False
                if (obs.aperture.AperName == alignment_reference_aperture_name) | (use_pseudo_fgs):
                    obs.aperture.plot(color='g', fill_color='0.7')
                elif obs.aperture.AperName == attitude_defining_aperture_name:
                    obs.aperture.plot(color='r', fill_color='0.5')
                else:
                    obs.aperture.plot()

                if (obs_collection.T['group_id'][i] == 0) & (obs_collection.T['INSTRUME'][i] != 'SUPERFGS'):
                    pl.plot(obs.star_catalog_matched['v2_spherical_arcsec'], obs.star_catalog_matched['v3_spherical_arcsec'], 'k.', ms=1, color='0.5')

                elif obs_collection.T['INSTRUME'][i] == 'SUPERFGS':
                    # this plots the matched stars as measured by the FGS into V2V3 an the FGS pickles
                    # the positions of stars reflect the latest TVS matrix?
                    pl.plot(obs.star_catalog_matched['v2_spherical_arcsec'], obs.star_catalog_matched['v3_spherical_arcsec'], \
                            marker=obs_collection.info_dict['visit_groups_parameters']['star_marker'][visit_group_index],
                            mec=obs_collection.info_dict['visit_groups_parameters']['color'][visit_group_index],
                            mfc='none',
                            ls='None',
                            ms=3)

                original_reference_position = np.array([obs.aperture.V2Ref, obs.aperture.V3Ref])
                plot_color = obs_collection.info_dict['visit_groups_parameters']['color'][visit_group_index]

                if (obs.observatory == 'HST') and ('FGS' in obs.aperture.AperName):
                    show_fgs_pickle_correction(obs.aperture, obs.aperture.tvs_corrected, factor=magnification_factor,
                                               show_original=False, color=plot_color, line_width=1)
                else:
                    offset_factor = +1
                    calibrated_offset = offset_factor*np.array([obs_collection.T['{}_v2_position_arcsec'.format(offset_specifier)][i], obs_collection.T['{}_v3_position_arcsec'.format(offset_specifier)][i]]) * magnification_factor
                    calibrated_uncertainty = np.array([obs_collection.T['sigma_{}_v2_position_arcsec'.format(offset_specifier)][i], obs_collection.T['sigma_{}_v3_position_arcsec'.format(offset_specifier)][i]]) * magnification_factor

                    calibrated_reference_position = original_reference_position + calibrated_offset
                    pl.plot(original_reference_position[0], original_reference_position[1], 'k+')
                    pl.plot(calibrated_reference_position[0], calibrated_reference_position[1], 'bo', mfc=plot_color, mec=plot_color)
                    pl.errorbar(calibrated_reference_position[0], calibrated_reference_position[1],
                                xerr=calibrated_uncertainty[0], yerr=calibrated_uncertainty[1], fmt='none',
                                ecolor=plot_color)

                    pl.plot([original_reference_position[0], calibrated_reference_position[0]],
                            [original_reference_position[1], calibrated_reference_position[1]], 'k--',
                            color='k')


            # plot HST FGS apertures
            if obs.observatory == 'HST':
                for fgs_name in ['FGS1', 'FGS2', 'FGS3']:
                    aperture = siaf[fgs_name]
                    aperture.plot(fill=False, color='k')
                    pl.text(aperture.V2Ref, aperture.V3Ref, fgs_name, horizontalalignment='center', zorder=50)

                for aperture_name in ['IUVIS1FIX', 'IUVIS2FIX', 'JWFC1FIX', 'JWFC2FIX']:
                    aperture = siaf[aperture_name]
                    # aperture.plot(fill=False, color='k')
                    pl.text(aperture.V2Ref-225, aperture.V3Ref+50, aperture_name, horizontalalignment='center', zorder=50)

            elif obs.observatory == 'JWST':
                for aperture_name, aperture in siaf.apertures.items():
                    if aperture_name not in unique_aperture_names_all:
                        aperture.plot(fill=False, color='k')
                        # pl.text(aperture.V2Ref, aperture.V3Ref, aperture_name, horizontalalignment='center', zorder=50)


            ax = pl.gca()
            if obs.observatory == 'HST':
                ax.invert_yaxis()
            pl.show()
            if save_plot == 1:
                figure_name = os.path.join(plot_dir, 'fpa_v2v3_offsets_apertures_{}_{}.pdf'.format(figure_filename_tag, offset_specifier))
                pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

            ############################
            # V2,V3 offsets summary plot


            aperture_groups = ['cameras']
            if obs.observatory == 'HST':
                if 'FGS' in ','.join(unique_aperture_names_all):
                    aperture_groups = ['cameras', 'guiders']
                # else:
                #     aperture_groups = ['cameras']
            else:
                unique_aperture_names = unique_aperture_names_all

            for aperture_group in aperture_groups:
                if obs.observatory == 'HST':
                    if aperture_group == 'cameras':
                        unique_aperture_names = np.array(
                            [s for s in unique_aperture_names_all if 'FGS' not in s])
                    elif aperture_group == 'guiders':
                        unique_aperture_names = np.array(
                            [s for s in unique_aperture_names_all if 'FGS' in s])

                figure_filename_tag = '{}_{}'.format(figure_filename_tag_orig, aperture_group)

                n_unique_apertures = len(unique_aperture_names)
                n_panels = n_unique_apertures
                n_figure_columns = n_unique_apertures
                n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

                fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                        figsize=(n_figure_columns * 4, n_figure_rows * 4),
                                        facecolor='w', edgecolor='k', sharex=True, sharey=True,
                                        squeeze=False)
                axes_max = 0
                for jj, aperture_name in enumerate(unique_aperture_names):
                    obs_index = np.where((obs_collection.T['AperName'] == aperture_name))[0]
                    fig_row = jj % n_figure_rows
                    fig_col = jj // n_figure_rows
                    axis = axes[fig_row][fig_col]

                    for epoch, visit_group in enumerate(visit_groups):
                        obs_visit_index = \
                        np.where(np.in1d(obs_collection.T['group_id'][obs_index], visit_group))[0]
                        plot_color = obs_collection.info_dict['visit_groups_parameters']['color'][epoch]
                        if 0:
                            axis.plot(obs_collection.T['{}_V2Ref'.format(offset_specifier)][obs_index][obs_visit_index].data,
                                      obs_collection.T['{}_V3Ref'.format(offset_specifier)][obs_index][obs_visit_index].data,
                                      'bo', mfc=plot_color, mec=plot_color)
                            axis.errorbar(obs_collection.T['{}_V2Ref'.format(offset_specifier)][obs_index][obs_visit_index],
                                          obs_collection.T['{}_V3Ref'.format(offset_specifier)][obs_index][obs_visit_index],
                                          xerr=obs_collection.T['sigma_{}_V2Ref'.format(offset_specifier)][obs_index][
                                              obs_visit_index],
                                          yerr=obs_collection.T['sigma_{}_V3Ref'.format(offset_specifier)][obs_index][
                                              obs_visit_index], fmt='none', ecolor=plot_color)
                        else:
                            axis.plot(obs_collection.T['{}_v2_position_arcsec'.format(offset_specifier)][obs_index][obs_visit_index].data,
                                      obs_collection.T['{}_v3_position_arcsec'.format(offset_specifier)][obs_index][obs_visit_index].data,
                                      'bo', mfc=plot_color, mec=plot_color)
                            axis.errorbar(obs_collection.T['{}_v2_position_arcsec'.format(offset_specifier)][obs_index][obs_visit_index],
                                          obs_collection.T['{}_v3_position_arcsec'.format(offset_specifier)][obs_index][obs_visit_index],
                                          xerr=obs_collection.T['sigma_{}_v2_position_arcsec'.format(offset_specifier)][obs_index][
                                              obs_visit_index],
                                          yerr=obs_collection.T['sigma_{}_v3_position_arcsec'.format(offset_specifier)][obs_index][
                                              obs_visit_index], fmt='none', ecolor=plot_color)
                        # if aperture_group == 'cameras':
                        if 1:
                            # show effect of attitude uncertainty on alignment results
                            attitude_pa_uncertainty_rad = np.array([obs_collection.observations[obs_index][jj].corrected_attitude['sigma_pa_arcsec_correction'].to(u.rad).value for jj in obs_visit_index])
                            v2 = np.array([obs_collection.observations[obs_index][jj].aperture.V2Ref for jj in obs_visit_index])
                            v3 = np.array([obs_collection.observations[obs_index][jj].aperture.V3Ref for jj in obs_visit_index])
                            radius_arcsec = np.linalg.norm([v2, v3], axis=0)
                            pa_nominal_rad = np.arctan2(v3, v2) # measured from V2-axis towards V3 axis
                            factors = np.array([-1, 1])
                            xys = [[radius_arcsec * np.cos(pa_nominal_rad+factor*attitude_pa_uncertainty_rad)-v2, radius_arcsec * np.sin(pa_nominal_rad+factor*attitude_pa_uncertainty_rad)-v3] for factor in factors]
                            # plot a kind of errorbar at the origin
                            axis.plot(np.hstack((xys[0][0], xys[1][0])), np.hstack((xys[0][1], xys[1][1])), 'k-', mfc=plot_color, mec=plot_color, lw=2)

                    title = aperture_name
                    if alignment_reference_aperture_name == aperture_name:
                        title += ' (reference)'
                    if 'IUVIS' in title:
                        title = 'WFC3 {}'.format(title)
                    elif 'JWFC' in title:
                        title = 'ACS {}'.format(title)

                    axis.set_title(title)
                    axis.axhline(y=0, color='0.5', ls='--', zorder=-50)
                    axis.axvline(x=0, color='0.5', ls='--', zorder=-50)
                    axis.set_xlabel('Offset in V2 (arcsec)')
                    axis.set_ylabel('Offset in V3 (arcsec)')
                    axis.grid()
                    tmp_axes_max = np.max(
                        np.abs(np.array([axis.get_xlim(), axis.get_ylim()]).flatten()))
                    if tmp_axes_max > axes_max:
                        axes_max = tmp_axes_max

                axis.set_xlim((-axes_max, axes_max))
                axis.set_ylim((-axes_max, axes_max))
                fig.tight_layout(h_pad=0.0)
                pl.show()
                if save_plot == 1:
                    figure_name = os.path.join(plot_dir, 'fpa_v2v3_offsets_summary{}_{}.pdf'.format(
                        figure_filename_tag, offset_specifier))
                    pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

            ############################
            # V2,V3 offsets summary plot as a function of time
            for aperture_group in aperture_groups:
                if obs.observatory == 'HST':
                    if aperture_group == 'cameras':
                        unique_aperture_names = np.array(
                            [s for s in unique_aperture_names_all if 'FGS' not in s])
                    elif aperture_group == 'guiders':
                        unique_aperture_names = np.array(
                            [s for s in unique_aperture_names_all if 'FGS' in s])

                figure_filename_tag = '{}_{}'.format(figure_filename_tag_orig, aperture_group)

                n_unique_apertures = len(unique_aperture_names)
                n_panels = n_unique_apertures * 2
                n_figure_columns = n_unique_apertures
                n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

                fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                        figsize=(n_figure_columns * 4, n_figure_rows * 2),
                                        facecolor='w', edgecolor='k', sharex=True, sharey=True,
                                        squeeze=False)
                axes_max = 0
                for jj, aperture_name in enumerate(unique_aperture_names):
                    obs_index = np.where((obs_collection.T['AperName'] == aperture_name))[0]
                    # fig_col = jj % n_figure_rows
                    fig_col = jj #// n_figure_rows


                    plot_time = Time(obs_collection.T['MJD'], format='mjd').decimalyear
                    # for ii, component in enumerate(['V2Ref', 'V3Ref']):
                    for ii, component in enumerate(['v2_position_arcsec', 'v3_position_arcsec']):
                        fig_row = ii
                        axis = axes[fig_row][fig_col]
                        axis.plot(plot_time[obs_index], obs_collection.T['{}_{}'.format(offset_specifier, component)][obs_index], 'ko')
                        axis.errorbar(plot_time[obs_index], obs_collection.T['{}_{}'.format(offset_specifier, component)][obs_index], yerr=obs_collection.T['sigma_{}_{}'.format(offset_specifier, component)][obs_index], fmt='none', ecolor='k')

                        title = aperture_name
                        if alignment_reference_aperture_name == aperture_name:
                            title += ' (reference)'
                        if 'IUVIS' in title:
                            title = 'WFC3 {}'.format(title)
                        elif 'JWFC' in title:
                            title = 'ACS {}'.format(title)

                        axis.axhline(y=0, color='0.5', ls='--', zorder=-50)
                        # axis.axvline(x=0, color='0.5', ls='--', zorder=-50)
                        if ii == 0:
                            axis.set_title(title)
                        else:
                            axis.set_xlabel('Time')
                        # if jj == 0:
                        #     axis.set_ylabel('Offset in {} (arcsec)'.format(component))

                        # axis.grid()
                    # tmp_axes_max = np.max(
                    #     np.abs(np.array([axis.get_xlim(), axis.get_ylim()]).flatten()))
                    # if tmp_axes_max > axes_max:
                    #     axes_max = tmp_axes_max

                # axis.set_xlim((-axes_max, axes_max))
                # axis.set_ylim((-axes_max, axes_max))
                fig.tight_layout(h_pad=0.0)
                pl.show()
                if save_plot == 1:
                    figure_name = os.path.join(plot_dir, 'fpa_v2v3_offsets_summary{}_{}_time.pdf'.format(
                        figure_filename_tag, offset_specifier))
                    pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

                # 1/0
                if offset_specifier == offset_specifiers[0]:
                    ############################
                    # plot scale deviations
                    n_panels = n_unique_apertures
                    n_figure_columns = n_unique_apertures
                    n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

                    fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                            figsize=(n_figure_columns * 4, n_figure_rows * 4),
                                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                                            squeeze=False)
                    axes_max = 0
                    for jj, aperture_name in enumerate(unique_aperture_names):
                        obs_index = np.where((obs_collection.T['AperName'] == aperture_name))[0]
                        fig_row = jj % n_figure_rows
                        fig_col = jj // n_figure_rows
                        axis = axes[fig_row][fig_col]
                        for epoch, visit_group in enumerate(visit_groups):
                            obs_visit_index = \
                            np.where(np.in1d(obs_collection.T['group_id'][obs_index], visit_group))[0]
                            plot_color = obs_collection.info_dict['visit_groups_parameters']['color'][epoch]

                            axis.plot(obs_collection.T['scale_in_x'][obs_index][obs_visit_index].data - 1,
                                      obs_collection.T['scale_in_y'][obs_index][obs_visit_index].data - 1,
                                      'bo', mfc=plot_color, mec=plot_color)
                            axis.errorbar(obs_collection.T['scale_in_x'][obs_index][obs_visit_index] - 1,
                                          obs_collection.T['scale_in_y'][obs_index][obs_visit_index] - 1,
                                          xerr=obs_collection.T['sigma_scale_in_x'][obs_index][
                                              obs_visit_index],
                                          yerr=obs_collection.T['sigma_scale_in_y'][obs_index][
                                              obs_visit_index], fmt='none', ecolor=plot_color)
                        title = aperture_name
                        if alignment_reference_aperture_name == aperture_name:
                            title += ' (reference)'
                        if 'IUVIS' in title:
                            title = 'WFC3 {}'.format(title)
                        elif 'JWFC' in title:
                            title = 'ACS {}'.format(title)

                        axis.set_title(title)
                        axis.axhline(y=0, color='0.5', ls='--', zorder=-50)
                        axis.axvline(x=0, color='0.5', ls='--', zorder=-50)
                        axis.set_xlabel('Scale offset from unity in X')
                        axis.set_ylabel('Scale offset from unity in Y')
                        axis.grid()
                        tmp_axes_max = np.max(
                            np.abs(np.array([axis.get_xlim(), axis.get_ylim()]).flatten()))
                        if tmp_axes_max > axes_max:
                            axes_max = tmp_axes_max
                    axis.set_xlim((-axes_max, axes_max))
                    axis.set_ylim((-axes_max, axes_max))
                    axis.xaxis.get_major_formatter().set_powerlimits((0, 1))
                    axis.yaxis.get_major_formatter().set_powerlimits((0, 1))
                    fig.tight_layout(h_pad=0.0)
                    pl.show()
                    if save_plot == 1:
                        figure_name = os.path.join(plot_dir, 'fpa_scale_offsets_summary{}.pdf'.format(
                            figure_filename_tag))
                        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

                    ############################
                    # plot skews
                    n_panels = n_unique_apertures
                    n_figure_columns = n_unique_apertures
                    n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

                    fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                            figsize=(n_figure_columns * 4, n_figure_rows * 4),
                                            facecolor='w', edgecolor='k', sharex=True, sharey=True,
                                            squeeze=False)
                    axes_max = 0
                    for jj, aperture_name in enumerate(unique_aperture_names):
                        obs_index = np.where((obs_collection.T['AperName'] == aperture_name))[0]
                        fig_row = jj % n_figure_rows
                        fig_col = jj // n_figure_rows
                        axis = axes[fig_row][fig_col]
                        for epoch, visit_group in enumerate(visit_groups):
                            obs_visit_index = \
                            np.where(np.in1d(obs_collection.T['group_id'][obs_index], visit_group))[0]
                            plot_color = obs_collection.info_dict['visit_groups_parameters']['color'][epoch]

                            axis.plot(obs_collection.T['on_axis_skew'][obs_index][obs_visit_index].data,
                                      obs_collection.T['off_axis_skew'][obs_index][obs_visit_index].data,
                                      'bo', mfc=plot_color, mec=plot_color)
                            axis.errorbar(obs_collection.T['on_axis_skew'][obs_index][obs_visit_index],
                                          obs_collection.T['off_axis_skew'][obs_index][obs_visit_index],
                                          xerr=obs_collection.T['sigma_on_axis_skew'][obs_index][
                                              obs_visit_index],
                                          yerr=obs_collection.T['sigma_off_axis_skew'][obs_index][
                                              obs_visit_index], fmt='none', ecolor=plot_color)
                        title = aperture_name
                        if alignment_reference_aperture_name == aperture_name:
                            title += ' (reference)'
                        if 'IUVIS' in title:
                            title = 'WFC3 {}'.format(title)
                        elif 'JWFC' in title:
                            title = 'ACS {}'.format(title)

                        axis.set_title(title)
                        axis.axhline(y=0, color='0.5', ls='--', zorder=-50)
                        axis.axvline(x=0, color='0.5', ls='--', zorder=-50)
                        axis.set_xlabel('On-axis skew (scale diff.)')
                        axis.set_ylabel('Off-axis skew (non-perpend.)')
                        axis.grid()
                        tmp_axes_max = np.max(
                            np.abs(np.array([axis.get_xlim(), axis.get_ylim()]).flatten()))
                        if tmp_axes_max > axes_max:
                            axes_max = tmp_axes_max
                    axis.set_xlim((-axes_max, axes_max))
                    axis.set_ylim((-axes_max, axes_max))
                    axis.xaxis.get_major_formatter().set_powerlimits((0, 1))
                    axis.yaxis.get_major_formatter().set_powerlimits((0, 1))
                    fig.tight_layout(h_pad=0.0)
                    pl.show()
                    if save_plot == 1:
                        figure_name = os.path.join(plot_dir, 'fpa_skew_summary{}.pdf'.format(
                            figure_filename_tag))
                        pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

                ############################
                # V3 clocking angle summary plot
                n_panels = n_unique_apertures
                n_figure_columns = n_unique_apertures
                n_figure_rows = np.int(np.ceil(n_panels / n_figure_columns))

                fig, axes = pl.subplots(n_figure_rows, n_figure_columns,
                                        figsize=(n_figure_columns * 4, n_figure_rows * 4),
                                        facecolor='w', edgecolor='k', sharex=True, sharey=True,
                                        squeeze=False)
                for jj, aperture_name in enumerate(unique_aperture_names):
                    obs_index = np.where((obs_collection.T['AperName'] == aperture_name))[0]
                    fig_row = jj % n_figure_rows
                    fig_col = jj // n_figure_rows
                    axis = axes[fig_row][fig_col]
                    for epoch, visit_group in enumerate(visit_groups):
                        obs_visit_index = \
                        np.where(np.in1d(obs_collection.T['group_id'][obs_index], visit_group))[0]
                        plot_color = obs_collection.info_dict['visit_groups_parameters']['color'][epoch]
                        if 0:
                            axis.plot(np.arange(len(obs_index))[obs_visit_index],
                                      obs_collection.T['{}_V3IdlYAngle'.format(offset_specifier)][obs_index][
                                          obs_visit_index].data * 3600, 'bo', mfc=plot_color, mec=plot_color)
                            axis.errorbar(np.arange(len(obs_index))[obs_visit_index],
                                          obs_collection.T['{}_V3IdlYAngle'.format(offset_specifier)][obs_index][
                                              obs_visit_index].data * 3600, yerr=
                                          obs_collection.T['sigma_{}_V3IdlYAngle'.format(offset_specifier)][obs_index][
                                              obs_visit_index] * 3600, fmt='none', ecolor=plot_color)
                        else:
                            abscissa = np.arange(len(obs_index))[obs_visit_index]
                            axis.plot(abscissa,
                                      obs_collection.T['{}_v3_angle_arcsec'.format(offset_specifier)][obs_index][
                                          obs_visit_index].data, 'bo', mfc=plot_color, mec=plot_color)
                            axis.errorbar(abscissa,
                                          obs_collection.T['{}_v3_angle_arcsec'.format(offset_specifier)][obs_index][
                                              obs_visit_index].data, yerr=
                                          obs_collection.T['sigma_{}_v3_angle_arcsec'.format(offset_specifier)][obs_index][
                                              obs_visit_index], fmt='none', ecolor=plot_color)

                        if aperture_group == 'guiders':
                            attitude_pa_uncertainty = [obs_collection.observations[obs_index][jj].corrected_attitude[
                                'sigma_pa_arcsec_correction'].to(u.arcsec).value for jj in obs_visit_index]
                            axis.errorbar(abscissa+0.2, abscissa*0., yerr=attitude_pa_uncertainty, fmt='none', ecolor=plot_color, elinewidth=2)#, alpha=0.5)

                            # 1/0

                    title = aperture_name
                    if alignment_reference_aperture_name == aperture_name:
                        title += ' (reference)'
                    if 'IUVIS' in title:
                        title = 'WFC3 {}'.format(title)
                    elif 'JWFC' in title:
                        title = 'ACS {}'.format(title)
                    axis.set_title(title)
                    axis.axhline(y=0, color='0.5', ls='--', zorder=-50)
                    axis.set_xlabel('Frame number')
                    axis.set_ylabel('V3IdlAngle correction (arcsec)')
                    pl.xlim((-0.5, len(obs_index) - 0.5))
                fig.tight_layout(h_pad=0.0)
                pl.show()
                if save_plot == 1:
                    figure_name = os.path.join(plot_dir, 'fpa_v3angle_offsets_summary{}_{}.pdf'.format(
                        figure_filename_tag, offset_specifier))
                    pl.savefig(figure_name, transparent=True, bbox_inches='tight', pad_inches=0)

    if random_realisation == 0:
        # obs_collection.T[
        #     'INSTRUME', 'CHIP', 'group_id', 'correction_V3IdlYAngle', 'sigma_calibrated_delta_V3IdlYAngle', 'correction_V2Ref', 'sigma_calibrated_delta_V2Ref', 'correction_V3Ref', 'sigma_calibrated_delta_V3Ref', 'number_of_used_stars_for_aperture_correction', 'number_of_matched_stars'].pprint()
        # , 'scale_in_x', 'sigma_scale_in_x', 'scale_in_y', 'sigma_scale_in_y'

        index = obs_collection.T['INSTRUME'] == 'SUPERFGS'
        obs_collection.T[
            'INSTRUME', 'APERTURE', 'group_id', 'number_of_measured_stars', 'number_of_gaia_stars', 'number_of_matched_stars', 'number_of_used_stars_for_aperture_correction'][
            index].pprint()


def evaluate_tvs(obs_collection, parameters, make_summary_figures=True, save_plot=True, plot_dir='', comparison_tvs_data=None, tvs_results={}, print_detailed_results=False):
    """

    Parameters
    ----------
    obs_collection
    parameters
    make_summary_figures
    save_plot
    plot_dir
    comparison_tvs_data
    tvs_results
    print_detailed_results

    Returns
    -------

    """
    # tvs_attributes = 'tvs_v2_arcsec tvs_v3_arcsec tvs_pa_deg tvs'.split()
    tvs_attributes = 'tvs_v2_arcsec tvs_v3_arcsec tvs_pa_deg'.split()
    tvs_results['tvs_attributes'] = tvs_attributes

    tvs_results['mc_results'] = {}
    random_realisation = obs_collection.info_dict['random_realisation']

    if 1:
    # if len(random_realisations) != 0:

        # print changes in FGS parameters and alignment/TVS matrix
        unique_aperture_names_all = np.unique(obs_collection.T['AperName'])
        fgs_aperture_names = [s for s in unique_aperture_names_all if 'FGS' in s]
        tvs_results['fgs_aperture_names'] = fgs_aperture_names
        for jj, aperture_name in enumerate(fgs_aperture_names):
            obs_indices = np.where((obs_collection.T['AperName'] == aperture_name))[0]
            tvs_results['mc_results'][aperture_name] = OrderedDict()
            for jjj, obs_index in enumerate(obs_indices):
                print('$'*50)
                # tvs_results = {}
                print('Aperture {} ; Date {}'.format(aperture_name, obs_collection.T['DATE-OBS'][obs_index]))
                print('Change relative to amu.rep version {}'.format(comparison_tvs_data['VERSION']))
                comparison_tvs_matrix = comparison_tvs_data[aperture_name.lower()]['tvs']

                if obs_collection.T['AperType'][obs_index] == 'PSEUDO':


                    offset_specifier = 'calibrated'
                    obs_collection.T[obs_index]
                    pseudo_aperture = copy.deepcopy(obs_collection.observations[obs_index].aperture)
                    for key in 'V2Ref V3Ref V3IdlYAngle'.split():
                        setattr(pseudo_aperture, '{}'.format(key),  obs_collection.T['{}_{}'.format(offset_specifier, key)][obs_index] )

                    # generate idl-tel coordinate pairs to apply SVD to determine updated TVS matrix
                    fgs_idl_x, fgs_idl_y = get_grid_coordinates(10, (0, -50), 1000, y_width=400)
                    fgs_idl_x += pseudo_aperture.idl_x_ref_arcsec
                    fgs_idl_y += pseudo_aperture.idl_y_ref_arcsec
                    method = 'planar_approximation'
                    input_coordinates = 'tangent_plane'
                    # output_coordinates = 'tangent_plane'
                    output_coordinates = 'spherical'
                    # 1/0
                    fgs_tel_v2, fgs_tel_v3 = pseudo_aperture.idl_to_tel(fgs_idl_x, fgs_idl_y, method=method,
                                                                 input_coordinates=input_coordinates,
                                                                 output_coordinates=output_coordinates)

                    x_idl_rad = np.deg2rad(fgs_idl_x / 3600.)
                    y_idl_rad = np.deg2rad(fgs_idl_y / 3600.)
                    z_idl_rad = np.sqrt(1. - x_idl_rad ** 2 - y_idl_rad ** 2)

                    idl_vector = np.vstack((x_idl_rad, y_idl_rad, z_idl_rad)).T

                    v2_tel = fgs_tel_v2 #np.array(obs.gaia_catalog_matched['v2_spherical_arcsec'])
                    v3_tel = fgs_tel_v3 #np.array(obs.gaia_catalog_matched['v3_spherical_arcsec'])

                    v2_tel_rad = np.deg2rad(v2_tel / 3600.)
                    v3_tel_rad = np.deg2rad(v3_tel / 3600.)
                    v1_tel_rad = np.sqrt(1. - v2_tel_rad ** 2 - v3_tel_rad ** 2)

                    tel_vector = np.vstack((v1_tel_rad, v2_tel_rad, v3_tel_rad)).T
                    if 0:
                        print('idl_vector norm {}'.format(np.linalg.norm(idl_vector, axis=1)))
                        print('tel_vector norm {}'.format(np.linalg.norm(tel_vector, axis=1)))

                    # solution using singular value decomposition
                    updated_tvs_matrix = rigid_transform_3D_rotation_only(np.mat(idl_vector), np.mat(tel_vector))

                    v = np.rad2deg(np.dot(updated_tvs_matrix, idl_vector.T))

                    v2_spherical_arcsec, v3_spherical_arcsec = v[1] * u.deg.to(u.arcsec), v[2] * u.deg.to(u.arcsec)
                    v2_new = v2_spherical_arcsec
                    v3_new = v3_spherical_arcsec

                    max_error_mas = np.max(np.linalg.norm([v2_new-fgs_tel_v2, v3_new-fgs_tel_v3], axis=0)) * 1000
                    rms_error_mas = np.std(np.linalg.norm([v2_new-fgs_tel_v2, v3_new-fgs_tel_v3], axis=0)) * 1000
                    print('{}: Maximum error of pseudo aperture idl_tel transformation across aperture: {:3.4f} mas'.format(aperture_name, max_error_mas))
                    print('{}: RMS of  error of pseudo aperture idl_tel transformation across aperture: {:3.4f} mas'.format(aperture_name, rms_error_mas))

                    if 0:
                        # check that residuals are around zero
                        print('SVD residuals {}'.format(
                            np.mean(np.dot(updated_tvs_matrix, idl_vector.T).T - tel_vector, axis=0)))

                    tvs_v2_arcsec, tvs_v3_arcsec, tvs_pa_deg, tvs = pseudo_aperture._tvs_parameters(updated_tvs_matrix)

                    obs_collection.observations[obs_index].aperture.corrected_tvs = tvs
                    attribute_values = {'tvs_v2_arcsec': tvs_v2_arcsec, 'tvs_v3_arcsec': tvs_v3_arcsec, 'tvs_pa_deg': tvs_pa_deg}
                    for attribute_name in tvs_attributes:
                        setattr(obs_collection.observations[obs_index].aperture, 'corrected_' + attribute_name, attribute_values[attribute_name])

                get_gsfc_delta_v(obs_collection.observations[obs_index].aperture.corrected_tvs,
                                 comparison_tvs_matrix, obs_collection.observations[obs_index].aperture)
                tvs_results['mc_results'][aperture_name][jjj] = get_gsfc_delta_v_mc(obs_collection.observations[obs_index].aperture, comparison_tvs_matrix, obs_collection.T[obs_index])

                if print_detailed_results:
                    for attribute_name in tvs_attributes:
                        if random_realisation == 0:
                            print('Original  {}: {}'.format(attribute_name, getattr(
                                obs_collection.observations[obs_index].aperture,
                                'db_' + attribute_name)))
                            print('Corrected {}: {}'.format(attribute_name, getattr(
                                obs_collection.observations[obs_index].aperture,
                                'corrected_' + attribute_name)))
                            tvs_results[
                                '{}_{}_original_epoch{}'.format(aperture_name, attribute_name, jjj)] = [
                                getattr(obs_collection.observations[obs_index].aperture,
                                        'db_' + attribute_name)]
                            tvs_results['{}_{}_corrected_epoch{}'.format(aperture_name, attribute_name,
                                                                         jjj)] = [
                                getattr(obs_collection.observations[obs_index].aperture,
                                        'corrected_' + attribute_name)]
                        else:
                            tvs_results['{}_{}_original_epoch{}'.format(aperture_name, attribute_name,
                                                                        jjj)].append([getattr(
                                obs_collection.observations[obs_index].aperture,
                                'db_' + attribute_name)])
                            tvs_results['{}_{}_corrected_epoch{}'.format(aperture_name, attribute_name,
                                                                         jjj)].append([getattr(
                                obs_collection.observations[obs_index].aperture,
                                'corrected_' + attribute_name)])

    return tvs_results


def flatten_list(seq):
    #from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
    if not seq:
        return []
    elif isinstance(seq[0],list):
        return (flatten_list(seq[0])+flatten_list(seq[1:]))
    else:
        return [seq[0]]+flatten_list(seq[1:])


def tvs_uncertainties(tvs_results, result_dir=''):
    """Produce estimates of TVS parameter uncertainties

    Parameters
    ----------
    tvs_results

    Returns
    -------

    """
    for epoch in range(len(tvs_results['visit_groups'])):
        print('TVS results for epoch {}'.format(epoch))

        tvs_result_tables = {}
        for jj, aperture_name in enumerate(tvs_results['fgs_aperture_names']):
            tvs_result_tables[aperture_name] = Table()
            for result_type in ['original', 'corrected']:
                for solution_name in ['full', 'bootstrap']:
                    mean_results = []
                    std_results = []
                    names = []
                    for attribute_name in tvs_attributes:
                        if attribute_name == 'tvs':
                            for ii in range(9):
                                names.append('tvs_m{}'.format(ii))
                        else:
                            names.append(attribute_name)

                        result_list = tvs_results[
                            '{}_{}_{}_epoch{}'.format(aperture_name, attribute_name, result_type,
                                                      epoch)]
                        # result_list = tvs_results['{}_{}_{}'.format(aperture_name, attribute_name, result_type)]
                        if solution_name == 'full':
                            result = np.asarray(result_list[0])
                        elif solution_name == 'bootstrap':
                            result = np.asarray(result_list[1:]).squeeze()

                        if (attribute_name == 'tvs') and (np.ndim(result) == 2):
                            mean_result = result.flatten()
                            std_result = np.zeros(mean_result.shape).flatten()
                        elif np.ndim(result) == 0:
                            mean_result = result
                            std_result = np.array([0])
                        else:
                            mean_result = np.mean(result, axis=0).flatten()
                            std_result = np.std(result, axis=0).flatten()

                        # print('{} {} {} {} mean={}, std={}'.format(aperture_name, attribute_name, result_type, solution_name,
                        #                                                mean_result, std_result))
                        if np.ndim(mean_result) != 0:
                            mean_results.append(mean_result.T.tolist())
                            std_results.append(std_result.T.tolist())
                        else:
                            mean_results.append(mean_result.tolist())
                            std_results.append(std_result.tolist())
                            # mean_results += (mean_result)
                            # std_results += (std_result)
                    # have to flatten the list before converting to table column

                    tvs_result_tables[aperture_name][
                        '{}_{}_mean'.format(result_type, solution_name)] = flatten_list(
                        mean_results)  # [item for sublist in mean_results for item in sublist]
                    tvs_result_tables[aperture_name][
                        '{}_{}_std'.format(result_type, solution_name)] = flatten_list(
                        std_results)  # [item for sublist in std_results for item in sublist]
            tvs_result_tables[aperture_name].add_column(Column(
                tvs_result_tables[aperture_name]['corrected_full_mean'] -
                tvs_result_tables[aperture_name]['original_full_mean'],
                name=('proposed_correction')))
            tvs_result_tables[aperture_name].add_column(Column(
                tvs_result_tables[aperture_name]['proposed_correction'] /
                tvs_result_tables[aperture_name]['original_full_mean'] * 100,
                name=('proposed_correction_percent')))
            tvs_result_tables[aperture_name].add_column(Column(names, name=('Parameter')), index=0)

            formats = {c: '%3.2e' for c in tvs_result_tables[aperture_name].colnames if
                       'Parameter' not in c}
            formats['proposed_correction_percent'] = '%3.2f'
            # tvs_result_tables[aperture_name].pprint()
            tvs_results_file = os.path.join(result_dir,
                                            'tvs_results_{}_epoch{}.csv'.format(aperture_name,
                                                                                epoch))
            tvs_result_tables[aperture_name][
                'Parameter     original_full_mean    corrected_full_mean  corrected_bootstrap_mean corrected_bootstrap_std proposed_correction proposed_correction_percent'.split()].write(
                tvs_results_file, format='ascii.fixed_width', bookend=False, delimiter=',',
                formats=formats, overwrite=True)


def generate_fgs_pseudo_aperture(siaf, show_plot=True, method='planar_approximation',
                                 use_tel_boresight=False, input_coordinates='tangent_plane',
                                 output_coordinates=None):
    """Generate apertures that realize the same idl_to_tel transformation as the FGS TVS matrix.

    Parameters
    ----------
    siaf

    Returns
    -------

    """

    siaf = copy.deepcopy(siaf)


    if show_plot:
        pl.figure()

    for aperture_name in siaf.apertures:
        if show_plot:
            siaf[aperture_name].plot(mark_ref=True, label=True)
        if 'FGS' in aperture_name:
            aperture = copy.deepcopy(siaf[aperture_name])
            # generate grid of ideal coordinates roughly centred at 0,0
            fgs_idl_x, fgs_idl_y = get_grid_coordinates(20, (0, -50), 1000, y_width=400)

            # offset grid to pickle position (FGS ideal (object space) coordinates span the HST FOV)
            fgs_idl_x += aperture.idl_x_ref_arcsec
            fgs_idl_y += aperture.idl_y_ref_arcsec

            # print(aperture_name)
            # print(aperture.idl_x_ref_arcsec, aperture.idl_y_ref_arcsec)
            # print(fgs_idl_x[0:10], fgs_idl_y[0:10])
            # continue

            # input coordinates for pseudo aperture transformation
            pseudo_fgs_idl_x = copy.deepcopy(fgs_idl_x)
            pseudo_fgs_idl_y = copy.deepcopy(fgs_idl_y)

            # transform to telescope coordinates
            fgs_tel_v2, fgs_tel_v3 = aperture.idl_to_tel(fgs_idl_x, fgs_idl_y,
                                                         method=method,
                                                         input_coordinates=input_coordinates,
                                                         output_coordinates='polar',
                                                         )

            if show_plot:
                pl.plot(fgs_tel_v2, fgs_tel_v3, 'b.')

            pseudo_aperture = copy.deepcopy(aperture)
            pseudo_aperture.AperType = 'PSEUDO'

            # use aperture correction code to generate pseudo-aperture that mimics the FGS
            # transformations using the classic formulae
            obs = AlignmentObservation('JWST', aperture_name)
            star_catalog = Table()
            star_catalog['x_idl_arcsec'], star_catalog['y_idl_arcsec'] = pseudo_fgs_idl_x, pseudo_fgs_idl_y
            star_catalog['star_id'] = np.arange(len(star_catalog))
            star_catalog['sigma_x_arcsec'] = np.ones(len(star_catalog))
            star_catalog['sigma_y_arcsec'] = np.ones(len(star_catalog))
            # reference catalog, this comes from the TVS transformation to tel frame
            gaia_catalog = Table()
            if method == 'spherical':
                gaia_catalog['v2_spherical_arcsec'], gaia_catalog['v3_spherical_arcsec'] = fgs_tel_v2, fgs_tel_v3
            gaia_catalog['source_id'] = np.arange(len(gaia_catalog))
            gaia_catalog['v2_error_arcsec'] = np.ones(len(gaia_catalog))
            gaia_catalog['v3_error_arcsec'] = np.ones(len(gaia_catalog))

            obs.star_catalog_matched = star_catalog
            obs.gaia_catalog_matched = gaia_catalog

            obs.number_of_matched_stars = len(gaia_catalog)

            # best guess
            pseudo_aperture.V2Ref = 0
            # pseudo_aperture.V3Ref = 0
            # pseudo_aperture.V3Ref = 0
            obs.aperture = copy.deepcopy(pseudo_aperture)
            if method == 'spherical':
                fieldname_dict = {}
                fieldname_dict['star_catalog'] = {}
                fieldname_dict['reference_catalog'] = {}

                # fieldname_dict['reference_catalog']['position_1'] = 'v2_tangent_arcsec'
                # fieldname_dict['reference_catalog']['position_2'] = 'v3_tangent_arcsec'
                fieldname_dict['reference_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['reference_catalog']['position_2'] = 'v3_spherical_arcsec'
                fieldname_dict['reference_catalog']['sigma_position_1'] = 'v2_error_arcsec'
                fieldname_dict['reference_catalog']['sigma_position_2'] = 'v3_error_arcsec'
                fieldname_dict['reference_catalog']['identifier'] = 'source_id'
                fieldname_dict['reference_catalog']['position_unit'] = u.arcsecond
                fieldname_dict['reference_catalog']['sigma_position_unit'] = u.arcsecond

                # fieldname_dict['star_catalog']['position_1'] = 'v2_tangent_arcsec'
                # fieldname_dict['star_catalog']['position_2'] = 'v3_tangent_arcsec'
                fieldname_dict['star_catalog']['position_1'] = 'v2_spherical_arcsec'
                fieldname_dict['star_catalog']['position_2'] = 'v3_spherical_arcsec'
                fieldname_dict['star_catalog']['sigma_position_1'] = 'sigma_x_arcsec'
                fieldname_dict['star_catalog']['sigma_position_2'] = 'sigma_y_arcsec'
                fieldname_dict['star_catalog']['identifier'] = 'star_id'
                fieldname_dict['star_catalog']['position_unit'] = u.arcsecond
                fieldname_dict['star_catalog']['sigma_position_unit'] = u.arcsecond

                obs.fieldname_dict = fieldname_dict
            else:
                obs.fieldname_dict = None


            determine_aperture_error(obs, obs, 0, 0, verbose=False,
                                                   plot_residuals=False, plot_dir='',
                                                   # plot_residuals=True, plot_dir='',
                                                   maximum_number_of_iterations=20,
                                                   use_fgs_pseudo_aperture=False,
                                                   use_tel_boresight=use_tel_boresight,
                                                   idl_tel_method=method,
                                                   fractional_threshold_for_iterations=0.0001,
                                                   k=8
                                                       )
            # pseudo_fgs_tel_v2, pseudo_fgs_tel_v3 = pseudo_aperture.idl_to_tel(pseudo_fgs_idl_x, pseudo_fgs_idl_y,
            #                                                                   V2Ref_arcsec=obs.aperture.V2Ref_corrected,
            #                                                                   V3Ref_arcsec=obs.aperture.V3Ref_corrected,
            #                                                                   V3IdlYAngle_deg=obs.aperture.V3IdlYAngle_corrected,
            #                                                                   method=method,
            #                                                                   input_coordinates=input_coordinates,
            #                                                                   output_coordinates=output_coordinates,
            #                                                                   )


            if (method=='planar_approximation') & 0:
                pseudo_fgs_tel_v2_s, pseudo_fgs_tel_v3_s = pseudo_aperture.idl_to_tel(pseudo_fgs_idl_x, pseudo_fgs_idl_y,
                                                                                  V2Ref_arcsec=pseudo_aperture.V2Ref_corrected,
                                                                                  V3Ref_arcsec=pseudo_aperture.V3Ref_corrected,
                                                                                  V3IdlYAngle_deg=pseudo_aperture.V3IdlYAngle_corrected,
                                                                                  method='spherical_transformation',
                                                                                  input_coordinates='tangent_plane')

                plot_aperture_names = ['FGS1', 'FGS2', 'FGS3', 'JWFCFIX', 'JWFC1FIX', 'JWFC2FIX', 'IUVISCTR',
                                       'IUVIS1FIX', 'IUVIS2FIX']
                data = {}
                data['reference'] = {'x': np.array(pseudo_fgs_tel_v2),
                                     'y': np.array(pseudo_fgs_tel_v3)}
                data['comparison_0'] = {'x': np.array(pseudo_fgs_tel_v2_s),
                                        'y': np.array(pseudo_fgs_tel_v3_s)}
                plot_spatial_difference(data, siaf=siaf, plot_aperture_names=plot_aperture_names)

                1/0


            # set alignment parameters of pseudo aperture
            for key in 'V2Ref V3Ref V3IdlYAngle'.split():
                setattr(pseudo_aperture, '{}'.format(key), getattr(obs.aperture, '{}_corrected'.format(key)))

            if method == 'spherical':
                # for pseudo aperture, input coordinates have to be spherical
                pseudo_fgs_tel_v2, pseudo_fgs_tel_v3 = pseudo_aperture.idl_to_tel(pseudo_fgs_idl_x,
                                                                                  pseudo_fgs_idl_y,
                                                                                  method=method,
                                                                                  input_coordinates='polar',
                                                                                  # input_coordinates='cartesian',
                                                                                  output_coordinates=None)


            else:
                pseudo_fgs_tel_v2, pseudo_fgs_tel_v3 = pseudo_aperture.idl_to_tel(pseudo_fgs_idl_x, pseudo_fgs_idl_y,
                                                                              method=method,
                                                                              input_coordinates=input_coordinates,
                                                                              output_coordinates=output_coordinates,
                                                                              )

            if 1:
                plot_aperture_names = ['FGS1', 'FGS2', 'FGS3', 'JWFCFIX', 'JWFC1FIX', 'JWFC2FIX', 'IUVISCTR', 'IUVIS1FIX', 'IUVIS2FIX']
                data = {}
                data['reference'] = {'x': np.array(fgs_tel_v2), 'y': np.array(fgs_tel_v3)}
                data['comparison_0'] = {'x': np.array(pseudo_fgs_tel_v2), 'y': np.array(pseudo_fgs_tel_v3)}
                plot_spatial_difference(data, siaf=siaf, plot_aperture_names=plot_aperture_names, make_new_figure=False)

            if show_plot:
                pl.plot(pseudo_fgs_tel_v2, pseudo_fgs_tel_v3, 'r.')
            max_error_mas = np.max(np.linalg.norm([pseudo_fgs_tel_v2-fgs_tel_v2, pseudo_fgs_tel_v3-fgs_tel_v3], axis=0)) * 1000
            rms_error_mas = np.std(np.linalg.norm([pseudo_fgs_tel_v2-fgs_tel_v2, pseudo_fgs_tel_v3-fgs_tel_v3], axis=0)) * 1000
            print('{}: Maximum error of pseudo aperture idl_tel transformation across aperture: {:3.4f} mas'.format(aperture_name, max_error_mas))
            print('{}: RMS of  error of pseudo aperture idl_tel transformation across aperture: {:3.4f} mas'.format(aperture_name, rms_error_mas))


            # setattr(siaf[aperture_name], 'pseudo_aperture', pseudo_aperture)
            setattr(siaf.apertures[aperture_name], 'pseudo_aperture', pseudo_aperture)


            # print(siaf[aperture_name].idl_to_tel(0., 0., method=method, input_coordinates=input_coordinates, output_coordinates=output_coordinates) )
            # print(siaf[aperture_name].pseudo_aperture.idl_to_tel(0., 0., method=method, input_coordinates=input_coordinates, output_coordinates=output_coordinates) )
            # print(siaf[aperture_name].idl_to_tel(0., 0.))
            # print(siaf[aperture_name].pseudo_aperture.idl_to_tel(0., 0.))
            # 1/0
            # assert (siaf[aperture_name].idl_to_tel(0., 0., method=method, input_coordinates=input_coordinates, output_coordinates=output_coordinates) == pseudo_aperture.idl_to_tel(0., 0., method=method, input_coordinates=input_coordinates, output_coordinates=output_coordinates))

            break

    if show_plot:
        pl.axis('tight')
        pl.axis('equal')
        pl.xlabel('V2 (arcsec)')
        pl.ylabel('V3 (arcsec)')
        ax = pl.gca()
        ax.invert_yaxis()
        pl.show()

    return siaf


def get_gsfc_v_parameters(tvs, aperture):
    """Return V1,V2,V3 in mas according to the definitions in E. Kimmer's reports.

    See Ed Kimmers email dated 20 July 2018:
    The way I compute the deltas is to extract the boresight vectors (third column in the tvs
    matrix) for both the new and reference (2011) matrices. This is based on an FGS coordinate
    definition that has the FGS boresight along the sensor Z axis.  I then take the difference
    between the second components to get the V2 displacement delta and the difference between the
    third components to get the V3 displacement delta. V2 and V3 are in vehicle space. The submatrix
    (2,1 to 3,2) represents the rotation about V1 so I extract the angle from there. But since this
    rotation is nearly about the FGS boresight I determine a typical displacement in the FGS field
    of view from the equation s=r*q = q ¸(1/ r), where “s” is the displacement in the field of
    view, “r” is the radial distance the point in the field of view, and  q  is the rotation
    angle about V1. The radial distance used is approximately the  center of the field of view
    (12.73 arcminutes is used as convenient value). Dividing the V1 rotation angle by  1/ r
    (~270) yields an approximate displacement in the center of the field of view and that is
    the delta I am reporting for V1.

    Parameters
    ----------
    tvs

    Returns
    -------

    """
    v2_rad = tvs[1, 2]
    v3_rad = tvs[2, 2]
    v1_arcsec, tmp1, tmp2, tmp3 = aperture._tvs_parameters(tvs=tvs, apply_rearrangement=False)
    v1_rad = -1 * v1_arcsec * u.arcsec.to(u.rad)
    # v1_rad /= 1 / (12.73 * u.arcmin.to(u.rad))
    v1_rad /= 1 / (12.02 * u.arcmin.to(u.rad))  # update in email from Ed Kimmer dated 2019-02-28

    conversion_factor = u.rad.to(u.milliarcsecond)

    v1 = v1_rad * conversion_factor
    v2 = v2_rad * conversion_factor
    v3 = v3_rad * conversion_factor

    return v1, v2, v3


def get_gsfc_delta_v(tvs, tvs_ref, aperture, verbose=True):
    """Return deltaV1,V2,V3 in mas according to the definitions in E. Kimmer's reports.

    Parameters
    ----------
    tvs
    tvs_ref

    Returns
    -------

    """

    v1_ref, v2_ref, v3_ref = get_gsfc_v_parameters(tvs_ref, aperture)
    v1, v2, v3 = get_gsfc_v_parameters(tvs, aperture)

    delta_v1 = v1 - v1_ref
    delta_v2 = v2 - v2_ref
    delta_v3 = v3 - v3_ref

    if verbose:
        print('Differences:')
        print('delta V1 {:>10.0f} mas'.format((v1 - v1_ref)))
        print('delta V2 {:>10.0f} mas'.format((v2 - v2_ref)))
        print('delta V3 {:>10.0f} mas'.format((v3 - v3_ref)))

    return delta_v1, delta_v2, delta_v3


def get_gsfc_delta_v_mc(aperture, tvs_ref, row, verbose=True):
    """Compute the GSFC offsets and account for uncertainties using MC.

    Parameters
    ----------
    aperture
    tvs_ref
    row : applicable row from the obs_collection summary table

    Returns
    -------

    """

    n_mc = 1000

    results = {}
    for flavour in ['calibrated_', 'corrected_']:
        results[flavour] = {}
        results[flavour]['attributes'] = []
        results[flavour]['alignment_parameters'] = {}
        results[flavour]['sigma_alignment_parameters'] = {}

        # 1/0
        for key, value in alignment_parameter_mapping['hst_fgs'].items():
            # attribute = '{}{}'.format(flavour, value)
            attribute = '{}{}_arcsec'.format(flavour, key)
            # alignment_parameters[key] = getattr(aperture, attribute)
            results[flavour]['alignment_parameters'][key] = row[attribute]
            results[flavour]['sigma_alignment_parameters'][key] = row['sigma_{}'.format(attribute)]
            results[flavour]['attributes'].append(attribute)

        results[flavour]['recomputed_tvs'] = aperture.compute_tvs_matrix(v2_arcsec=results[flavour]['alignment_parameters']['v2_position'],
                                           v3_arcsec=results[flavour]['alignment_parameters']['v3_position'],
                                           pa_deg=results[flavour]['alignment_parameters']['v3_angle']/3600.)

        tvs_mc = np.zeros((3, 3, n_mc))
        delta_v_mc = np.zeros((3, n_mc))

        seed = 1234567
        np.random.seed(seed)
        v2_arcsec_mc = results[flavour]['alignment_parameters']['v2_position'] \
                       + np.random.normal(0., results[flavour]['sigma_alignment_parameters']['v2_position'], n_mc)
        np.random.seed(seed+1)
        v3_arcsec_mc = results[flavour]['alignment_parameters']['v3_position'] \
                       + np.random.normal(0., results[flavour]['sigma_alignment_parameters']['v3_position'], n_mc)
        np.random.seed(seed+2)
        v3_angle_mc = results[flavour]['alignment_parameters']['v3_angle'] \
                       + np.random.normal(0., results[flavour]['sigma_alignment_parameters']['v3_angle'], n_mc)

        for j in range(n_mc):
            tvs_mc[:,:,j] = aperture.compute_tvs_matrix(v2_arcsec=v2_arcsec_mc[j],
                                               v3_arcsec=v3_arcsec_mc[j],
                                               pa_deg=v3_angle_mc[j]/3600.)

            delta_v_mc[:, j] = get_gsfc_delta_v(tvs_mc[:,:,j], tvs_ref, aperture, verbose=False)
        if verbose:
            print('{} TVS offsets ({} Monte Carlo sets)'.format(flavour, n_mc))
        for j in range(3):
            results[flavour]['delta_v{}_mas'.format(j+1)] = np.mean(delta_v_mc[j, :])
            results[flavour]['sigma_delta_v{}_mas'.format(j+1)] = np.std(delta_v_mc[j, :])
            if verbose:
                print('MC: delta V{} {:>6.0f} +/- {:>3.0f} (rms) mas'.format(j+1, results[flavour]['delta_v{}_mas'.format(j+1)], results[flavour]['sigma_delta_v{}_mas'.format(j+1)])
                                                                              )
        results[flavour]['mean_tvs'] = np.mean(tvs_mc, axis=2)
        results[flavour]['sigma_mean_tvs'] = np.std(tvs_mc, axis=2)

    results['DATE-OBS'] = row['DATE-OBS']
    return results


def get_v1_rotation(tvs, aperture):

    v2, v3, v1, tvs = aperture._tvs_parameters(tvs=tvs)
    v2_rad = -1 * v2 * u.arcsec.to(u.rad)

    v2_rad /= 1 / (12.73 * u.arcmin.to(u.rad))
    return v2_rad


def show_fgs_pickle_correction(aperture, tvs_corrected, factor=200., show_original=False,
                               new_figure=False, color='r', show_zero_point=False,
                               show_fiducial_offset=True, line_width=2,
                               apply_factor_to='offsets'):
    """Show the effect of a TVS matrix update.

    Parameters
    ----------
    aperture_name
    tvs_corrected
    factor

    Returns
    -------

    """
    if 'fgs' not in aperture.AperName.lower():
        raise ValueError

    if new_figure:
        pl.figure()
    if show_original:
        aperture.plot(color='b')

    # original pickle outline in FGS object space
    fgs_idl_x, fgs_idl_y = aperture.closed_polygon_points('idl')

    tvs_parameters_original = aperture._tvs_parameters()
    tvs_parameters_corrected = aperture._tvs_parameters(tvs=tvs_corrected)

    if apply_factor_to == 'parameters':
        tvs_parameters_difference = [a-b for a,b in zip(tvs_parameters_corrected,tvs_parameters_original)]

        # exaggerate offsets
        tvs_parameters = [b+factor*d for b,d in zip(tvs_parameters_original, tvs_parameters_difference)]

    elif apply_factor_to == 'offsets':
        tvs_parameters = tvs_parameters_corrected

    fgs_tel_v2_s, fgs_tel_v3_s = aperture.idl_to_tel(fgs_idl_x, fgs_idl_y,
                                                     V3IdlYAngle_deg=tvs_parameters[2],
                                                     V2Ref_arcsec=tvs_parameters[0],
                                                     V3Ref_arcsec=tvs_parameters[1])


    if apply_factor_to == 'offsets':
        fgs_tel_v2_original, fgs_tel_v3_original = aperture.idl_to_tel(fgs_idl_x, fgs_idl_y)
        # apply magnification factor to offsets in V frame
        fgs_tel_v2_s = fgs_tel_v2_original + factor*(fgs_tel_v2_s - fgs_tel_v2_original)
        fgs_tel_v3_s = fgs_tel_v3_original + factor*(fgs_tel_v3_s - fgs_tel_v3_original)



    pl.plot(fgs_tel_v2_s, fgs_tel_v3_s, color, lw=line_width)
    if show_zero_point:
        if apply_factor_to == 'parameters':
            pl.plot(*aperture.idl_to_tel(0, 0), 'x', color='b', mew=line_width)

            pl.plot(*aperture.idl_to_tel(0,0,V3IdlYAngle_deg=tvs_parameters[2],
                                                         V2Ref_arcsec=tvs_parameters[0],
                                                         V3Ref_arcsec=tvs_parameters[1]), 'r+', mec=color, mew=line_width)

        elif apply_factor_to == 'offsets':
            original_zero_point = aperture.idl_to_tel(0, 0)
            corrected_zero_point = aperture.idl_to_tel(0, 0, V3IdlYAngle_deg=tvs_parameters[2],
                                                         V2Ref_arcsec=tvs_parameters[0],
                                                         V3Ref_arcsec=tvs_parameters[1])
            corrected_zero_point = original_zero_point + factor*(corrected_zero_point-original_zero_point)
            pl.plot(*original_zero_point, 'x', color='b', mew=line_width)
            pl.plot(*corrected_zero_point, 'r+', mec=color, mew=line_width)

    if show_fiducial_offset:
        idl_x_ref_arcsec, idl_y_ref_arcsec =  aperture.tel_to_idl(aperture.V2Ref, aperture.V3Ref)
        original_reference_position = np.array(aperture.idl_to_tel(idl_x_ref_arcsec, idl_y_ref_arcsec))
        calibrated_reference_position = np.array(aperture.idl_to_tel(idl_x_ref_arcsec, idl_y_ref_arcsec,
                                                            V3IdlYAngle_deg=tvs_parameters[2],
                                                            V2Ref_arcsec=tvs_parameters[0],
                                                            V3Ref_arcsec=tvs_parameters[1]))
        if 0:
            print(aperture)
            print(aperture.V2Ref, aperture.V3Ref)
            print(original_reference_position)
            print(calibrated_reference_position)

        if apply_factor_to == 'offsets':
            calibrated_reference_position = original_reference_position + factor*(calibrated_reference_position-original_reference_position)




        pl.plot(original_reference_position[0], original_reference_position[1], 'k+', mec=color)

        pl.plot(calibrated_reference_position[0], calibrated_reference_position[1], 'bo',
                mfc=color, mec=color)

        pl.plot([original_reference_position[0], calibrated_reference_position[0]],
                [original_reference_position[1], calibrated_reference_position[1]], 'k--', color='k')

    if new_figure:
        pl.axis('equal')
        pl.show()
