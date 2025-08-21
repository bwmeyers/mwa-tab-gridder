#!/usr/bin/env python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from astropy.coordinates import SkyCoord, SkyOffsetFrame
import astropy.units as u
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle


def hex_grid_tangent_plane(
    center: SkyCoord,
    fov_deg: float = 0.5,
    overlap_frac: float = 0.2,
    rings: int = 1,
) -> list[SkyCoord]:

    # Define a local tangent-plane coordinate frame centered on `center`
    tangent_frame = SkyOffsetFrame(origin=center)

    # Hex grid setup in tangent plane (offsets in degrees)
    spacing = fov_deg * (1 - overlap_frac)  # allow beam overlap
    dx = spacing  # horizontal spacing
    dy = spacing * np.sqrt(3) / 2  # vertical spacing

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
                angle = np.pi / 3 * i
                for j in range(r):
                    # Compute hex steps
                    x = (r * np.cos(angle) - j * np.cos(angle + np.pi / 3)) * dx
                    y = (r * np.sin(angle) - j * np.sin(angle + np.pi / 3)) * dy

                    # Create offset in tangent plane
                    offset = SkyCoord(x * u.deg, y * u.deg, frame=tangent_frame)

                    # Transform back to RA/Dec and append to pointing list
                    pointings.append(offset.transform_to("icrs"))

    return pointings


def plot_pointings_with_projection(
    pointings: list[SkyCoord],
    projection_center: SkyCoord = None,
    fov_deg: float = 1.0,
) -> None:

    # If there's no preferred projection centre, just use the first position in the list
    if projection_center is None:
        projection_center = pointings[0]

    # Setup WCS for gnomonic projection (TAN)
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
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=wcs)
    ax.set_title("Sky Pointings with in Gnomonic (TAN) Projection")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    # Plot beams
    for p in pointings:
        circle = SphericalCircle(
            (p.ra, p.dec),
            (fov_deg / 2) * u.deg,  # radius
            edgecolor="blue",
            facecolor="none",
            transform=ax.get_transform("world"),
            alpha=0.6,
        )
        ax.add_patch(circle)

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

    x_min = np.min(x_pix) - 50
    x_max = np.max(x_pix) + 50
    y_min = np.min(y_pix) - 50
    y_max = np.max(y_pix) + 50

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.grid(color="gray", ls="dotted")
    plt.show()


if __name__ == "__main__":
    # Example parameters
    center = SkyCoord(ra=180 * u.deg, dec=-60 * u.deg, frame="icrs")
    fov_major = 0.5  # degrees
    fov_minor = 0.5  # degrees
    overlap = 0.2  # 20% overlap
    n_rings = 2  # 2 rings = 19 total beams
    pa = 0  # ellipse rotated 30 deg east of north

    grid_points = hex_grid_tangent_plane(
        center, min((fov_major, fov_minor)), overlap, n_rings
    )
    plot_pointings_with_projection(grid_points, fov_deg=min((fov_major, fov_minor)))

    for gp in grid_points:
        print(f"{gp.to_string('hmsdms', sep=':', pad=True, precision=3)}")
