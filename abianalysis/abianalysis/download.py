#  MIT License
#
#  Copyright (c) 2022. Stan Kerstjens
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
"""

Downloads
=========

Download the data from the ABI.

"""

import io
import logging
from typing import Iterable, Tuple
import zipfile
import json

import requests
import numpy as np

from volume.volume import Volume, AGES


def _paged(query, max_num_rows=50) -> Iterable:
    """Iterate through an ABI RMA query.

    The first page will always have a single row. From this first page, the
    total number of rows is read. From the second to the penultimate page, each
    page will have max_num_rows. The last page will have however many rows are
    left.

    :param str query:
        The query to the ABI RMA. The start_row and num_rows filters will be
        appended to the query.

    :param int max_num_rows:
        The maximum number of rows in a page.

    """
    start_row = 0

    # Only an initial value, total rows will be reset as soon as the first
    # query result comes in
    total_rows = start_row + 1

    session = requests.Session()

    while start_row < total_rows:
        num_rows = min(max_num_rows, total_rows - start_row)

        rows = '[start_row$eq%d][num_rows$eq%d]' % (start_row, num_rows)
        response = session.get(query + rows).json()
        if response['success']:
            total_rows = response['total_rows']

            logging.info("[%6d-%6d]/%6d: loading rows...",
                         start_row,
                         start_row + num_rows,
                         total_rows)

            yield response['msg']

            start_row += num_rows
        else:
            logging.error("no successful response")

    session.close()


def _dim_size(mhd) -> Tuple[int, int, int]:
    """Retrieve the dimensions (shape) of the array from the mhd file

    :returns: (x, y, z) dimensions of the array
    """
    for byte_line in mhd.readlines():
        line = byte_line.decode()
        var, val = [s.strip() for s in line.split('=')]

        if var == 'DimSize':
            x, y, z = (int(v.strip()) for v in val.split(' '))
            return x, y, z


def download_genes() -> Iterable[Tuple[int, str]]:
    """Retrieve all genes in the Developing Mouse Atlas of the ABI.

    Generates tuples containing the ABI id of the gene, and the gene acronym.

    """
    # noinspection SpellCheckingInspection
    query = ("http://api.brain-map.org/api/v2/data/query.json?"
             "criteria=model::Gene,"
             "rma::criteria,products[abbreviation$eqDevMouse],"
             "rma::options"
             "[only$eq'genes.id,genes.acronym']")

    for data in _paged(query):
        for datum in data:
            yield int(datum['id']), datum['acronym']


def expression_meta():
    """Retrieve all meta-data for the volumes pertaining to the developmental
    mouse atlas.

    This includes only sagittal sections of the main ages (E11.5, E13.5, E15.5,
    E18.5, P4, P14, P28, and P56. All experiments marked 'failed' are excluded.
    Duplicate experiments at the same age for the same gene are not removed,
    and appear as duplicates in the data base.

    Generates tuples containing (SectionDataId, gene id, age, donor id).

    The donor id is the ABI id of the mouse used in the experiment.

    """
    # noinspection SpellCheckingInspection,HttpUrlsUsage
    query = ("http://api.brain-map.org/api/v2/data/query.json?"
             "criteria=model::SectionDataSet,"

             # No failed experiments
             "rma::criteria,[failed$eqfalse],"

             # Only the sagittal data set
             "[data_sets.plane_of_section_id$eq2],"

             # Only the major ages
             "[ages.name$in'E11.5','E13.5','E15.5','E18.5',"
             "'P4','P14','P28','P56'],"

             # Only the developing mouse database
             "products[abbreviation$eqDevMouse],"

             # Only include a subset of the available information, namely the
             #  - id of the section
             #  - id of the gene
             #  - age of the donor
             #  - id of the donor
             "rma::include,specimen(donor(age)),genes,"
             "rma::options[only$eq'"
             "data_sets.id,"
             "specimens.,"
             "donors.id,"
             "ages.name,"
             "genes.id,"
             "']")

    for data in _paged(query):
        for datum in data:
            if len(datum['genes']) >= 1:
                id_ = datum['id']
                age = datum['specimen']['donor']['age']['name']
                donor_id = datum['specimen']['donor']['id']

                # There will always only be a single gene
                # associated with a valid volume, but in the ABI
                # database this is still implemented with a list.
                # Hence the [0].
                gene_id = datum['genes'][0]['id']
                yield id_, gene_id, age, donor_id


def _expression_data(id_) -> np.ndarray:
    """Download expression data for the specified SectionDataId

    Expression is downloaded to the data folder in the form of a {id_}.npy
    file. This is uncompressed.

    :param int id_:
        SectionDataId

    """
    query = 'http://api.brain-map.org/grid_data/download/%d?include=energy'

    response = requests.get(query % id_)

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    with archive.open('energy.mhd', 'r') as mhd:
        shape = _dim_size(mhd)[::-1]

    with archive.open('energy.raw', 'r') as arr_file:
        data = np.frombuffer(arr_file.read(),
                             dtype=np.float32).reshape(shape)

    return data


def _find_gene(ids, shape):
    # First we try to find any of the saved volumes by trying to load
    # it. if it is successful we break the loop and go to the next
    # gene.
    for id_ in ids:
        file_name = f'resources/expression/{id_}.npy'
        try:
            np.load(file_name)
        except IOError:
            pass
        else:
            break

    # If the for loop runs without breaking, we have not found a saved
    # volume and we will proceed trying to download one.  If the
    # download is successful (i.e. we can load the saved volume) we
    # break and go to the next gene.  If no volume can be downloaded,
    # we issue an error log and move on,
    else:
        for id_ in ids:
            try:
                exp = _expression_data(id_)
                if exp.shape == shape:
                    file_name = f'resources/expression/{id_}.npy'
                    np.save(file_name, exp)
                else:
                    logging.debug('mismatch')
            except IOError:
                pass
            else:
                break
        else:
            raise RuntimeError('Gene not found')


def _load_genes():
    genes_file = 'resources/genes.json'
    try:
        with open(genes_file, 'r') as file_handle:
            genes = json.load(file_handle)
    except IOError:
        genes = dict(download_genes())
        with open(genes_file, 'w') as file_handle:
            json.dump(genes, file_handle)
    return genes


def _load_meta(genes):
    meta_file = 'resources/expression.json'
    try:
        with open(meta_file, 'r') as file_handle:
            meta = json.load(file_handle)
    except IOError:
        meta = {
            age: {
                gene_id: [] for gene_id in genes
            } for age in AGES
        }

        for id_, gene_id, age, _ in expression_meta():
            meta[age][gene_id].append(id_)

        with open(meta_file, 'w') as file_handle:
            json.dump(meta, file_handle)
    return meta


def _load_anatomy_grids():
    return {
        age: np.load(f'resources/anatomy/{age}.npy')
        for age in AGES
    }


def _load_nissl_grids():
    return {
        age: np.load(f'resources/nissl/{age}.npy')
        for age in AGES
    }


def _download_expression(genes, meta, anatomy_grids):
    for age in AGES:
        shape = anatomy_grids[age].shape

        # We will try to find a single volume per gene.  As soon as we find
        # one, we stop.
        i = 0
        for gene_id in genes:
            try:
                _find_gene(meta[age][gene_id], shape)
            except RuntimeError:
                logging.debug('Gene not found %s', gene_id)
            else:
                i += 1

        logging.info('%s: found %d / %d genes', age, i, len(genes))


def _construct_volumes(genes, meta, anatomy_grids):
    for age in AGES:
        anatomy = anatomy_grids[age]
        positions = np.argwhere(anatomy > 0)
        mask = tuple(map(tuple, positions.T))
        expression = []
        acronyms = []

        logging.info('Constructing volume %s...', age)

        for gene_id in genes:
            for id_ in meta[age][gene_id]:
                file_name = f'resources/expression/{id_}.npy'
                try:
                    expression.append(np.load(file_name)[mask])
                except IOError:
                    pass
                else:
                    acronyms.append(genes[gene_id])
                    break

        logging.info(' - %d voxels.', len(expression[0]))
        logging.info(' - %d / %d genes.', len(acronyms), len(genes))

        volume = Volume(
            expression=np.vstack(expression).T,
            voxel_indices=positions,
            genes=acronyms,
            anatomy=anatomy[mask],
            age=age,
        )

        volume.save(f'resources/volumes/{age}.npz')


def main():
    """Download meta and expression data, and load them into volumes"""

    logging.basicConfig(level=logging.INFO)

    genes = _load_genes()
    meta = _load_meta(genes)
    anatomy_grids = _load_anatomy_grids()

    _download_expression(genes, meta, anatomy_grids)
    _construct_volumes(genes, meta, anatomy_grids)


if __name__ == '__main__':
    main()
