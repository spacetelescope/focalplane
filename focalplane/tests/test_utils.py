import os

from astropy.table import Table
from astropy.time import Time
from astroquery.gaia import Gaia

from ..utils import correct_for_proper_motion

ON_TRAVIS = os.environ.get('TRAVIS') == 'true'

local_dir = os.path.dirname(os.path.abspath(__file__))

@pytest.mark.skipif(ON_TRAVIS, reason='timeout issue.')
def test_pm_correction():
    query = """SELECT * FROM gaiadr2.gaia_source AS gaia WHERE gaia.parallax > 200 AND gaia.parallax < 205"""

    output_file = os.path.join(local_dir, 'nearby_sources.vot'.format())
    overwrite = True

    if (not os.path.isfile(output_file)) or (overwrite):
        job = Gaia.launch_job_async(query, dump_to_file=True, output_file=output_file)
        table = job.get_results()
    else:
        table = Table.read(output_file)
    print('Retrieved {} sources'.format(len(table)))

    target_epoch = Time(2017, format='jyear')
    corrected_table = correct_for_proper_motion(table, target_epoch)

    assert len(corrected_table) == len(table)