#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Sets all the common figure specifications.
"""

# Standard library

# Third party requirements

# Local imports

# Constants
specs = {
    'fontsize_title':           11,
    'fontsize_ticks':           11,
    'fontsize_label':           11,
}


def set_specs(
        ax,
        fig_size=None,

        x_lim=None,
        x_ticks=None,

        y_lim=None,
        y_ticks=None,

        aspects=None,
    ):
    """ Set the specifications of the figure.

    Args:
        ax (Axes):
        fig_size (iterable): Size of figure (in inches)
        x_lim (list): Limit of x-axis
        x_ticks (list): X-ticks
        y_lim (list): Limit of x-axis
        y_ticks (list): Y-ticks
        aspects (list of str): Aspects as 'equal', 'box', etc.

    Returns:
        None

    """

    # Font sizes
    ax.tick_params(
        labelsize=specs['fontsize_ticks'],
    )

    xlabel = ax.xaxis.get_label().get_text()
    ylabel = ax.yaxis.get_label().get_text()
    if xlabel is not None and xlabel is not '':
        ax.set_xlabel(xlabel, fontsize=specs['fontsize_label'])
    if ylabel is not None and ylabel is not '':
        ax.set_ylabel(ylabel, fontsize=specs['fontsize_label'])

    im = ax.images
    if len(im) > 0:
        cbar = im[-1].colorbar
        cbar.ax.tick_params(labelsize=specs['fontsize_ticks'])

    if fig_size is not None:
        ax.figure.set_size_inches(*fig_size)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    if aspects is not None:
        ax.set_aspect(*aspects)
