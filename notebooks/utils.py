# Utilities for caching
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Dropdown, VBox
from joblib import Memory
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
from pythreejs import Points, BufferGeometry, BufferAttribute, PointsMaterial, \
    PerspectiveCamera, Scene, Renderer, OrbitControls
from sklearn.preprocessing import minmax_scale

from abianalysis.hierarchy import make_hierarchy, Hierarchy
from abianalysis.plot.hierarchy_plotter import HierarchyPlotter
from abianalysis.volume import AGES, Volume
from abianalysis.volume.preprocess import match_genes

cache_dir = 'cache'
memory = Memory(cache_dir, verbose=0)


@memory.cache()
def get_gene_list() -> List[str]:
    volumes = [Volume.load(a) for a in AGES]
    for volume in volumes:
        volume.preprocess()
    return match_genes(*volumes)


@memory.cache()
def load_volume(age: str, shuffled: bool = False) -> Volume:
    volume = Volume.load(age)
    volume.preprocess()
    genes = get_gene_list()
    volume.filter_genes(np.isin(volume.genes, np.array(genes)))
    if shuffled:
        volume.shuffle()
    return volume


@memory.cache()
def load_hierarchy(age: str, shuffled: bool = False) -> Hierarchy:
    volume = load_volume(age, shuffled=shuffled)
    hierarchy = make_hierarchy(volume)
    return hierarchy


def gene_viewer(volume):
    gene_dropdown = Dropdown(
        options=sorted(get_gene_list()),
        description='Gene:'
    )

    svv = SimpleVolumeViewer(
        positions=volume.voxel_indices,
        values=volume.expression[:, 0])

    def update_gene(change):
        gene = list(volume.genes).index(gene_dropdown.value)
        svv.values = volume.expression[:, gene]

    gene_dropdown.observe(update_gene)
    return VBox([gene_dropdown, svv.renderer])


class SimpleVolumeViewer:
    def __init__(self, positions, values, point_size=2, width=500, height=500):
        self.cmap = 'viridis'
        colors = get_cmap(self.cmap)(minmax_scale(values))
        self.points = Points(
            geometry=BufferGeometry(
                attributes={
                    'position': BufferAttribute(
                        array=positions.astype(np.float32)),
                    'color': BufferAttribute(
                        array=colors.astype(np.float32))
                }
            ),
            material=PointsMaterial(vertexColors='VertexColors',
                                    size=point_size),
            position=tuple(-np.mean(positions, 0))
        )
        self.camera = PerspectiveCamera(position=[-100, 0, 0], up=[0, -1, 0],
                                        aspect=width / height, near=50)
        self.scene = Scene(children=[self.points])
        self.renderer = Renderer(
            camera=self.camera, scene=self.scene,
            alpha=False,
            controls=[OrbitControls(controlling=self.camera, zoomSpeed=.1)],
            width=width,
            height=height,
        )

    @property
    def positions(self):
        return self.points.geometry.attributes['position'].array

    @positions.setter
    def positions(self, positions):
        self.points.geometry.attributes['position'] = BufferAttribute(
            array=positions.astype(np.float32))
        self.points.position = tuple(-np.mean(positions, 0))

    @property
    def colors(self):
        return self.points.geometry.attributes['color'].array

    @property
    def values(self):
        return None

    @values.setter
    def values(self, values):
        colors = get_cmap(self.cmap)(minmax_scale(values))
        self.points.geometry.attributes['color'].array = colors.astype(
            np.float32)


def get_row_height(size):
    return size[0] + size[1]


def get_width_ratios(size):
    return [get_row_height(size), size[0] + size[2]]


def get_main_gridspec(max_depth, size, **kwargs):
    return GridSpec(nrows=max_depth, ncols=4,
                    left=0, right=1, bottom=0, top=1,
                    width_ratios=[1, 2, 1, 2],
                    height_ratios=[*[3] * max_depth],
                    hspace=2 / 5, **kwargs)



def get_depth_vs_spatial_spread(hierarchy):
    # Every node in the hierarchy corresponds to a collection of voxels.
    max_depth = max(node.depth for node in hierarchy.leaves())
    depth_vs_spatial_spread = {i: [] for i in range(max_depth + 1)}
    for node in hierarchy.descendants():
        depth_vs_spatial_spread[node.depth].append(node.spatial_spread())
    return depth_vs_spatial_spread


def plot_region_sizes(ax, hierarchy):
    depths, region_sizes = zip(*get_depth_vs_spatial_spread(hierarchy).items())
    means = np.array([np.mean(r) for r in region_sizes]) / 1000
    std = np.array([np.std(r) for r in region_sizes]) / 1000
    ax.fill_between(depths, means - std, means + std, facecolor='lightgrey')
    ax.plot(depths, means, c='k')


def make_plot(hierarchy, shuffled, max_depth=3):
    helper = HierarchyPlotter(hierarchy)
    sh_helper = HierarchyPlotter(shuffled)
    fig = plt.figure(dpi=75, figsize=(8, 10))
    fig.text(0, 1, hierarchy.volume.age, fontweight='bold')

    gs = get_main_gridspec(max_depth, helper.image_size)

    for depth in range(max_depth):
        show_labels = depth == 0

        ax = fig.add_subplot(gs[depth, 0])
        helper.plot_expression(ax, depth + 1, show_labels=show_labels)
        helper.plot_views(fig, gs[depth, 1], depth + 1, show_labels=show_labels)

        ax = fig.add_subplot(gs[depth, 2])
        sh_helper.plot_expression(ax, depth + 1, show_labels=show_labels)
        sh_helper.plot_views(fig, gs[depth, 3], depth + 1,
                             show_labels=show_labels)

    return fig
