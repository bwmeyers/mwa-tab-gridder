#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
import astropy.constants as c
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle, add_scalebar

from mwalib import MetafitsContext, Pol


def hex_grid_tangent_plane(
    center: SkyCoord,
    fov: u.Quantity["angle"] | None = None,
    overlap_frac: float = 0.2,
    rings: int = 1,
) -> list[SkyCoord]:

    # Define a default for the FoV in case it is not provided.
    if fov == None:
        fov = 0.5 * u.deg

    # Define a local tangent-plane coordinate frame centered on `center`
    tangent_frame = SkyOffsetFrame(origin=center)

    # Hex grid setup in tangent plane (offsets in degrees)
    spacing = 2 * fov.to(u.deg) * (1 - overlap_frac)  # allow beam overlap
    dx = spacing  # horizontal spacing in degrees
    dy = spacing * np.sqrt(3) / 2  # vertical spacing in degrees
    # TODO: Probably need to figure out how to pack elliptical beams with arb rotation, etc....

    # Loop over concentric hex rings
    pointings = []
    for r in range(0, rings + 1):
        if r == 0:
            # Create offset in tangent plane
            offset = SkyCoord(0 * u.deg, 0 * u.deg, frame=tangent_frame)

            # Transform back to RA/Dec and append to pointing list
            pointings.append(offset.transform_to("icrs"))
        else:
            for i in range(6):  # 6 sides of hexagon
                angle = (np.pi / 3) * i
                for j in range(r):
                    # Compute hex steps
                    x = (r * np.cos(angle) - j * np.cos(angle + np.pi / 3)) * dx
                    y = (r * np.sin(angle) - j * np.sin(angle + np.pi / 3)) * dy

                    # Create offset in tangent plane
                    offset = SkyCoord(x, y, frame=tangent_frame)

                    # Transform back to RA/Dec and append to pointing list
                    pointings.append(offset.transform_to("icrs"))

    return pointings


def fwhm(
    freq_hz: u.Quantity["frequency"],
    max_baseline: u.Quantity["length"],
    scale: str | None = "airy",
) -> u.Quantity["angle"]:
    """Convert a provided observing frequency and maximum array baseline to a nominal FWHM."""

    if scale == "airy":
        scale = 1.22 * u.radian
    else:
        scale = 1 * u.radian

    wl = c.c / freq_hz
    th = (scale * wl / max_baseline).decompose().to(u.radian)
    return th


def plot_pointings_with_projection(
    pointings: list[SkyCoord],
    fov: u.Quantity["angle"] | None = None,
    wcs_config: WCS | dict | None = None,
) -> None:

    if fov == None:
        fov = 0.5 * u.deg

    if isinstance(wcs_config, dict):
        wcs = WCS(wcs_config)
    elif isinstance(wcs_config, WCS):
        wcs = wcs_config
    else:
        # Setup WCS for gnomonic projection (TAN)
        projection_center = pointings[0]
        pixel_scale = 0.01  # deg/pixel = 36 arcsec/pixel
        image_size = 1000  # pixels
        wcs_dict = {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRVAL1": projection_center.ra.deg,
            "CRVAL2": projection_center.dec.deg,
            "CRPIX1": image_size / 2,
            "CRPIX2": image_size / 2,
            "CD1_1": -pixel_scale,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": pixel_scale,
            "NAXIS1": image_size,
            "NAXIS2": image_size,
        }
        wcs = WCS(wcs_dict)

    # Prepare figure
    fig = plt.figure(figsize=plt.figaspect(0.9), constrained_layout=True)
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_title("Sky Pointings with in Gnomonic (TAN) Projection")
    ax.set_xlabel("Right Ascension (J2000)")
    ax.set_ylabel("Declination (J2000)")

    # Plot beams
    # TODO: Allow elliptical beams to be plotted?

    for p in pointings:
        circle = SphericalCircle(
            (p.ra, p.dec),
            fov.to(u.deg),  # radius
            edgecolor="blue",
            facecolor="none",
            transform=ax.get_transform("world"),
            alpha=0.6,
        )
        ax.add_patch(circle)
        ax.scatter(
            x=p.ra,
            y=p.dec,
            s=20,
            color="blue",
            transform=ax.get_transform("world"),
        )
    add_scalebar(
        ax,
        length=fov.to(u.arcmin),
        label=f"{fov.to(u.arcmin).value:.1f}'",
    )

    # Auto-adjust plot limits based on beam positions
    ra_vals = np.array([p.ra.deg for p in pointings])
    dec_vals = np.array([p.dec.deg for p in pointings])

    # Convert corners to pixel coords using WCS
    corners_world = SkyCoord(
        [ra_vals.min(), ra_vals.max(), ra_vals.min(), ra_vals.max()] * u.deg,
        [dec_vals.min(), dec_vals.min(), dec_vals.max(), dec_vals.max()] * u.deg,
        frame="icrs",
    )
    x_pix, y_pix = wcs.world_to_pixel(corners_world)
    pad = 1.2 * fov.to(u.deg).value / pixel_scale

    x_min = np.min(x_pix) - pad
    x_max = np.max(x_pix) + pad
    y_min = np.min(y_pix) - pad
    y_max = np.max(y_pix) + pad

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.grid(color="gray", ls="dotted")
    plt.show()


def find_maximum_mwa_baseline(
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


def mwa_gridder_cli() -> None:
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
        required=True,
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

    generate_mwa_grid(parser)


def generate_mwa_grid(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    # Get basic array configuration
    if args.metafits is not None:
        mwa_context = MetafitsContext(args.metafits)
        freq_hz = mwa_context.centre_freq_hz * u.Hz
        max_baseline = find_maximum_mwa_baseline(mwa_context)[0]

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
    mwa_gridder_cli()
