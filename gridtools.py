#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
import astropy.constants as c
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle, add_scalebar


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
