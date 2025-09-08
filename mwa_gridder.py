#!/usr/bin/env python

import argparse
import numpy as np
from scipy.spatial.distance import cdist

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

from mwalib import MetafitsContext, Pol

from gridtools import hex_grid_tangent_plane, plot_pointings_with_projection, fwhm


def find_maximum_baseline(
    context: MetafitsContext,
) -> tuple[u.Quantity["length"], np.ndarray[u.Quantity["length"]]]:
    """From the observation metadata, compute the tile maximum baseline."""
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    tile_positions = np.delete(tile_positions, np.where(tile_flags & True), axis=0)

    dist = cdist(tile_positions, tile_positions)
    dist = np.delete(dist.flatten(), np.where(dist.flatten() <= 0.01))  # remove autos

    return max(dist) * u.m, dist * u.m


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mwa_gridder",
        description="""
        A tool to calculate tied-array beam pointing directions in an 
        organised centered hexagonal grid around a provided central point.
        """,
        epilog="""
        NOTE: The tied-array beam FWHM can often be asymmetric, 
        especially far from zenith. In those cases it is best to either increase 
        the overlap fraction. 

        (In future releases, we will add the ability to simulate the tied-array 
        beam shape, and the FWHM will be set conservatively to be the radius of 
        the inscribing circle.)
        """,
    )
    parser.add_argument("-m", "--metafits", type=str, help="MWA Metafits file.")
    parser.add_argument(
        "--freq",
        type=float,
        help="Observing frequency (Hz). Overrides what is in provided metafits file.",
    )
    parser.add_argument(
        "--bmax",
        type=float,
        help="Maximum baseline (m) during observation. Overrides what is in provided metafits file.",
    )
    parser.add_argument(
        "-c",
        "--center",
        type=str,
        help="""The J2000 Right Ascension (hh:mm:ss) and Declination (±dd:mm:ss) 
        for the center of the grid. (Enter as a single string, space delimited, i.e., 
          'hh:mm:ss ±dd:mm:ss' 
        to ensure correct parsing.)""",
    )
    parser.add_argument(
        "-n",
        "--nrings",
        type=int,
        help="Number of concentric hexagonal rings to produce.",
        default=1,
    )
    parser.add_argument(
        "-o",
        "--overlap",
        type=float,
        help="""The minimum overlap fraction of beam FWHM. 
        (A negative number adds unfilled space between beam pointings.)""",
        default=0.2,
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Toggle writing computed pointing centres to file 'pointings.txt'",
        default=False,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot of computed pointings on the sky plane.",
        default=False,
    )
    args = parser.parse_args()

    # Get basic array configuration
    if args.metafits is not None:
        mwa_context = MetafitsContext(args.metafits)
        freq_hz = mwa_context.centre_freq_hz * u.Hz
        max_baseline = find_maximum_baseline(mwa_context)[0]

    # Overrides, if provides
    if args.freq:
        freq_hz = args.freq * u.Hz
    if args.bmax:
        max_baseline = args.bmax * u.m

    fov = fwhm(freq_hz, max_baseline)
    print(f"Maximum baseline, B = {max_baseline:g}")
    print(f"Centre frequency, f = {freq_hz.to(u.MHz):g}  λ = {(c.c/freq_hz).to(u.m):g}")
    print(f"FWHM ~ 1.22λ/B ~ {fov.to(u.deg):g} = {fov.to(u.arcmin):g}")
    center = SkyCoord(
        f"{args.center.split(' ')[0]}",
        f"{args.center.split(' ')[1]}",
        unit=("hourangle", "deg"),
        frame="icrs",
    )
    overlap = args.overlap
    n_rings = args.nrings
    # Number of pointings = 1 + 6*N*(N-1)/2 total beams (centered-hexagonal numbers, one-based)
    n_pts = 1 + 6 * (n_rings + 1) * (n_rings) // 2  # zero-based

    print(
        f"Generating centred hexagonal grid with {n_rings} concentric rings = {n_pts} pointings"
    )
    grid_points = hex_grid_tangent_plane(center, fov, overlap, n_rings)
    if args.show:
        plot_pointings_with_projection(grid_points, fov=fov)

    if not args.write:
        for gp in grid_points:
            print(f"{gp.to_string('hmsdms', sep=':', pad=True, precision=3)}")
    else:
        with open("pointings.txt", "w") as fh:
            for gp in grid_points:
                fh.write(f"{gp.to_string('hmsdms', sep=':', pad=True, precision=3)}\n")


if __name__ == "__main__":
    main()
