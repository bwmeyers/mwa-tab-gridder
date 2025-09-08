# MWA tied-array beam gridding tool

This tool aims to help the user figure out a reasonably good centered hexagonal layout of tied-array beams to form around a central point. 
Overlaps of adjacent beams is supported, and the utility can accept a MWA "metafits" file to help determine array configuration details which are used to space
the tied-array beams on the sky.

The tesselation is done in the sky tangent plane, thus projection effects are completely accounted for throughout. 

## Installation
The package is setup in a "flat layout" (so everything is within a single Python file/module), thus a simple 

```pip install .``` (within the source directory) 

or 

```pip install git+https://github.com/bwmeyers/mwa-tab-gridder.git```

should do the trick. 

_As per usual, it is recommended to do so in a virtual environment._

## Usage
After installation, the executable (entry point) will be `mwa-tab-grid`, e.g.,
```
> mwa-tab-grid -h
usage: mwa_gridder [-h] [-m METAFITS] [--freq FREQ] [--bmax BMAX] -c CENTER [-n NRINGS] [-o OVERLAP] [--write] [--show]
...
```
where each option has help text and sensible defaults, where applicable. 

- The `--freq` and `--bmax` options will each override quantities derived from the MWA metafits file. Both argument should be provided if no metafits file is provided as input.
- The `--write` option will write the calculated tied-array beam positions to `pointings.txt` in the current working directory. 
- The `--show` option will produce a standard `matplotlib` plotting window with the tied-array beam positions (and nominal widths) displayed in the TAN projection on WCS axes. 
