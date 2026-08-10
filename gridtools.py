
import argparse

import arviz as az
import astropy.constants as c
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from astropy.visualization.wcsaxes import SphericalCircle, add_scalebar
from astropy.wcs import WCS
from mwalib import MetafitsContext, Pol
from scipy.spatial.distance import cdist


def hex_grid_tangent_plane(
    center: SkyCoord,
    fov: u.Quantity["angle"] | None = None,
    overlap_frac: float = 0.2,
    rings: int = 1,
) -> list[SkyCoord]:

    # Define a default for the FoV in case it is not provided.
    if fov is None:
        fov = 0.5 * u.deg
    else:
        fov = fov.to(u.deg)

    # Define a local tangent-plane coordinate frame centered on `center`
    tangent_frame = SkyOffsetFrame(origin=center)

    # Hex grid setup in tangent plane (offsets in degrees)
    spacing = 2 * fov * (1 - overlap_frac)  # allow beam overlap
    dx = spacing  # horizontal spacing in degrees
    dy = spacing * np.sqrt(3) / 2  # vertical spacing in degrees
    # TODO: Probably need to figure out how to pack elliptical beams
    # with arb rotation, etc....

    # Loop over concentric hex rings
    pointings = []
    for r in range(rings + 1):
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
    """Convert a provided observing frequency and maximum array baseline
    to a nominal FWHM."""

    if scale == "airy":
        scale = 1.22 * u.radian
    else:
        scale = 1 * u.radian

    wl = c.c / freq_hz
    th = (scale * wl / max_baseline).decompose()
    return th


def plot_pointings_with_projection(
    pointings: list[SkyCoord],
    fov: u.Quantity["angle"] | None = None,
    wcs_config: WCS | dict | None = None,
) -> None:

    if fov is None:
        fov = 0.5 * u.deg
    else:
        fov = fov.to(u.deg)

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
    ax.set_title("Sky Pointings in Gnomonic (TAN) Projection")
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
            p.ra,
            p.dec,
            s=20,
            color="blue",
            transform=ax.get_transform("world"),
        )
    add_scalebar(
        ax,
        length=fov.to(u.arcmin),
        label=f"{fov.to(u.arcmin).value:.1f}'",
    )

    ax.grid(color="gray", ls="dotted")
    plt.savefig("pointing_grid.png", dpi=200, bbox_inches="tight")


def find_characteristic_baseline(
    context: MetafitsContext,
    hdi_prob: float = 0.9,
    extra_tile_flags: list[str] | None = None,
    exclude_flagged: bool = True,
) -> tuple[float, np.ndarray, float, np.ndarray]:
    """From the observation metadata, compute the tile effective and
    maximum baselines, as well as the baseline distribution.

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        hdi_prob (float, optional): Fraction of baselines to be included for the
            highest-density interval. Defaults to 0.9.
        extra_tile_flags (list[str] | None, optional): A list of additional
        tile names to flag as bad. Defaults to None.
        exclude_flagged (bool, optional): Whether to exclude flagged tiles
            from the baseline distribution.
    Returns:
        tuple[float, float, np.ndarray, np.ndarray]: A tuple containing:
            (1) The baseline mode (i.e., the most common baseline length),
            (2) The maximum baseline,
            (3) The highest-density interval, and
            (4) The baseline distribution.
    """
    tile_positions = np.array(
        [
            np.array([rf.east_m, rf.north_m, rf.height_m])
            for rf in context.rf_inputs
            if rf.pol == Pol.X
        ]
    )
    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if rf.tile_name in extra_tile_flags or str(rf.tile_id) in extra_tile_flags:
                tile_flags[itile] = True
            itile += 1

    if exclude_flagged:
        tile_positions = np.delete(
            tile_positions,
            np.where(tile_flags & True),
            axis=0,
        )

    dist = cdist(tile_positions, tile_positions)
    dist = np.delete(dist.flatten(), np.where(dist.flatten() <= 0.01))  # remove autos
    max_dist = np.max(dist) * u.m
    distances = dist * u.m

    # use a KDE approach to estimate the mode of the baseline distribution
    grid, density, _ = az.kde(dist)
    dist_mode = grid[np.argmax(density)] * u.m
    dist_hdi = np.asarray(az.hdi(dist, prob=hdi_prob, method="nearest")) * u.m

    return dist_mode, max_dist, dist_hdi, distances


def plot_baseline_distribution(
    context: MetafitsContext,
    hdi_prob: float = 0.9,
    extra_tile_flags: list[str] | None = None,
    show_flagged_tiles: bool = True,
) -> None:
    """Plot the baseline distribution and indicate the highest-density interval(s).

    Args:
        context (MetafitsContext): A mwalib.MetafitsContext object that contains the
            array configuration and delay settings.
        extra_tile_flags (list[str] | None, optional): A list of additional
            tile names to flag as bad. Defaults to None.
        show_flagged_tiles (bool): Plot the flagged tiles in a different colour.
            Default: True.
    """
    _, max_baseline, hdi_baseline, baselines = find_characteristic_baseline(
        context,
        hdi_prob=hdi_prob,
        extra_tile_flags=extra_tile_flags,
        exclude_flagged=show_flagged_tiles,
    )
    eff_baseline = np.max(hdi_baseline)

    tile_flags = np.array([rf.flagged for rf in context.rf_inputs if rf.pol == Pol.X])
    if extra_tile_flags is not None:
        itile = 0
        for rf in context.rf_inputs:
            if rf.pol != Pol.X:
                continue
            if rf.tile_name in extra_tile_flags or str(rf.tile_id) in extra_tile_flags:
                tile_flags[itile] = True
            itile += 1

    num_ok_tiles = (~tile_flags).sum()
    num_bad_tiles = (tile_flags).sum()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    ax.hist([b.value for b in baselines], bins=np.arange(0, max_baseline.value, 10))
    ymax = max(ax.get_ylim())

    if len(np.shape(hdi_baseline)) > 1:
        for i in list(hdi_baseline):
            ax.fill_between(i.value, 0, ymax, color="0.8", alpha=0.5)
    else:
        ax.fill_between(
            [h.value for h in hdi_baseline],
            0,
            ymax,
            color="0.8",
            alpha=0.5,
        )
    ax.axvline(eff_baseline.value, ls=":", color="k")
    ax.text(
        x=0.95,
        y=0.95,
        s=f"Number of baselines = {len(baselines)}\n"
        + f"Number of 'good' tiles = {num_ok_tiles}\n"
        + f"Number of flagged tiles = {num_bad_tiles}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=12,
    )
    plt.xlim(0, None)
    plt.ylim(None, ymax)
    plt.xlabel("Baseline length (m)", fontsize=14)
    plt.ylabel("Frequency of baseline length", fontsize=14)
    plt.title(
        f"Observation ID: {context.obs_id}  ({context.sched_start_utc})\n"
        + rf"Max. baseline $\approx$ {max_baseline * u.m:.0f}  "
        + rf"Characteristic baseline $\approx$ {eff_baseline:.0f}"
    )
    plt.minorticks_on()
    plt.tick_params(labelsize=12)
    plt.savefig(f"{context.obs_id}_baseline_dist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def mwa_gridder_cli() -> None:
    parser = argparse.ArgumentParser(
        prog="mwa_gridder",
        description="""
        A tool to calculate tied-array beam pointing directions in an
        organised centered hexagonal grid around a provided central point.
        """,
        epilog="""
        NOTE: The tied-array beam FWHM can often be asymmetric,
        especially far from zenith. In those cases it is best to either
        increase the overlap fraction.

        (In future releases, we will add the ability to simulate the tied-array
        beam shape, and the FWHM will be set conservatively to be the radius of
        the inscribing circle.)
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--metafits", type=str, help="MWA Metafits file.")
    parser.add_argument(
        "-f",
        "--freq",
        type=float,
        help="Observing frequency (Hz). Overrides what is in provided metafits file.",
    )
    parser.add_argument(
        "-B",
        "--bmax",
        type=float,
        help="Maximum baseline (m) during observation. Overrides what is in "
        "provided metafits file.",
    )
    parser.add_argument(
        "-c",
        "--center",
        type=str,
        help="""The J2000 Right Ascension (hh:mm:ss) and
        Declination (±dd:mm:ss) for the center of the grid.
        (Enter as a single string, space delimited, i.e.,
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
        "--use-simple-fov",
        action="store_true",
        help="Use a naive FWHM = 1.22λ/Bmax approximation.",
        default=False,
    )
    parser.add_argument(
        "--eff-baseline-frac",
        type=float,
        help="""Calculate the effective baseline by ensuring this fraction of
        baselines are captured within the highest density interval.""",
        default=0.90,
    )
    parser.add_argument(
        "--tile-flags",
        type=str,
        help="A comma-separated list of tile names or IDs to flag.",
        default=None,
    )
    generate_mwa_grid(parser)


def generate_mwa_grid(parser: argparse.ArgumentParser):
    args = parser.parse_args()

    # Get basic array configuration
    if args.metafits is not None:
        mwa_context = MetafitsContext(args.metafits)
        freq_hz = mwa_context.centre_freq_hz * u.Hz
        char_bline, max_bline, hdi_bline, _ = find_characteristic_baseline(
            mwa_context,
            hdi_prob=args.eff_baseline_frac,
            extra_tile_flags=args.tile_flags,
        )
        plot_baseline_distribution(
            mwa_context,
            hdi_prob=args.eff_baseline_frac,
            extra_tile_flags=args.tile_flags,
        )
        eff_bline = hdi_bline.max()
        # Take the maximum of the HDI (highest density interval) as the
        # effective array Bmax, as it represents that
        # (eff_baseline_frac * 100)% of baselines are equal or less than
        # this value, i.e., short baselines dominate while preserving the
        # fact that there are sufficient baselines of length > max(HDI) which
        # will act to increase the effective array Bmax.

    # Overrides, if provides
    if args.freq:
        freq_hz = args.freq * u.Hz
    if args.bmax:
        eff_bline = args.bmax * u.m
        char_bline = eff_bline
        max_bline = eff_bline

    if args.use_simple_fov:
        fov = fwhm(freq_hz, max_bline, scale="airy")
    else:
        fov = fwhm(freq_hz, eff_bline, scale=None)
    print(f"Maximum baseline, Bmax = {max_bline:g}")
    print(f"Approx. mode of baselines = {char_bline:g}")
    print(f"Effective baseline, Beff = {eff_bline:g}")
    print(
        f"Centre frequency, f = {freq_hz.to(u.MHz):g}  λ = {(c.c / freq_hz).to(u.m):g}"
    )
    if args.use_simple_fov:
        print(f"FWHM ~ 1.22λ/Bmax ~ {fov.to(u.deg):g} = {fov.to(u.arcmin):g}")
    else:
        print(f"FWHM ~ λ/Beff ~ {fov.to(u.deg):g} = {fov.to(u.arcmin):g}")

    center = SkyCoord(
        f"{args.center.split(' ')[0]}",
        f"{args.center.split(' ')[1]}",
        unit=("hourangle", "deg"),
        frame="icrs",
    )
    overlap = args.overlap
    # Since the TAB shape becomes more elongated as the centre moves away from zenith
    # the elongation affects the ratio of major/minor axes as ~1/sin(el)
    if args.metafits is not None:
        overlap0 = overlap
        k = 0.8
        alt = mwa_context.alt_rad
        overlap = overlap0 + k * (1 - np.sin(alt))
        print("Adjusting overlap to approx. account for project effects")
        print(
            f"    New overlap = {overlap0} + {k} * (1 - sin({mwa_context.alt_deg:g})) = {overlap}"
        )

        if overlap > 0.75:
            print("Restricting overlap to 0.75")
            overlap = 0.75

    n_rings = args.nrings
    # Number of pointings = 1 + 6*N*(N-1)/2 total beams
    # (centered-hexagonal numbers, one-based)
    n_pts = 1 + 6 * (n_rings + 1) * (n_rings) // 2  # zero-based

    print(
        f"Generating centred hexagonal grid with {n_rings} concentric rings = "
        f"{n_pts} pointings"
    )
    grid_points = hex_grid_tangent_plane(center, fov, overlap, n_rings)

    plot_pointings_with_projection(grid_points, fov=fov)

    if not args.write:
        for gp in grid_points:
            print(f"{gp.to_string('hmsdms', sep=':', pad=True, precision=3)}")
    else:
        with open("pointings.txt", "w") as fh:
            for gp in grid_points:
                gp_str = gp.to_string("hmsdms", sep=":", pad=True, precision=3)
                fh.write(f"{gp_str}\n")


if __name__ == "__main__":
    mwa_gridder_cli()
