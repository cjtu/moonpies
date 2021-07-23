"""Setup matplotlib with custom installed fonts and user defaults."""
import sys
import shutil
from pathlib import Path
from matplotlib import get_cachedir, get_configdir, matplotlib_fname

# Downloads path is default to search for new fonts
DOWNLOADS_DIR = Path.home().joinpath('Downloads')

# Define the rcparams to use in matplotlib
# See https://matplotlib.org/stable/tutorials/introductory/customizing.html
RC_STR = """
#### CUSTOM MATPLOTLIBRC FORMAT
# See https://matplotlib.org/stable/tutorials/introductory/customizing.html

## LINES
# See https://matplotlib.org/api/artist_api.html#module-matplotlib.lines

## PATCHES                                                                 
# See https://matplotlib.org/api/artist_api.html#module-matplotlib.patches

## HATCHES                                                                 

## BOXPLOT                                                                 

## FONT                                                                    
# The font properties used by `text.Text`.
# See https://matplotlib.org/api/font_manager_api.html for more information

font.family:  sans-serif
font.size:    12.0
font.sans-serif: Myriad Pro, Helvetica, DejaVu Sans, Arial, sans-serif

## TEXT                                                                    
# The text properties used by `text.Text`.
# See https://matplotlib.org/api/artist_api.html#module-matplotlib.text

## LaTeX                                                                   
# See https://matplotlib.org/tutorials/text/usetex.html

## AXES                                                                    
# See https://matplotlib.org/api/axes_api.html#module-matplotlib.axes
axes.facecolor:     white   # axes background color
axes.grid:          True   # display grid or not
axes.formatter.limits: -3, 3  # use scientific notation if log10
                            # of the axis range is smaller than the
                            # first or larger than the second
#axes.prop_cycle: cycler('color', ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf'])
                # color cycle for plot lines as list of string color specs:
                # single letter, long name, or web-style hex
                # As opposed to all other parameters in this file, the color
                # values must be enclosed in quotes for this parameter,
                # e.g. '1f77b4', instead of 1f77b4.
                # See also https://matplotlib.org/tutorials/intermediate/color_cycle.html
                # for more details on prop_cycle usage.
axes.formatter.min_exponent: 3  # minimum exponent to use scientific notation 

## AXIS                                                                    

## DATES                                                                   

## TICKS                                                                   
# See https://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
xtick.top:           True   # draw ticks on the top side
xtick.bottom:        True    # draw ticks on the bottom side
xtick.direction:     in     # direction: {in, out, inout}

ytick.left:          True    # draw ticks on the left side
ytick.right:         True   # draw ticks on the right side
ytick.direction:     in     # direction: {in, out, inout}

## GRIDS                                                                   

## LEGEND                                                                  

## FIGURE                                                                  
# See https://matplotlib.org/api/figure_api.html#matplotlib.figure.Figure
#figure.figsize:    7.4, 5.55  # figure size in inches
figure.facecolor:   white     # figure face color
figure.frameon:     True      # enable figure frame

## Figure layout
figure.autolayout: True  # When True, automatically adjust subplot
                            # parameters to make the plot fit the figure
                            # using `tight_layout`

## IMAGES                                                                  
image.interpolation: 'none'  # see help(imshow) for options
image.cmap:   viridis              # A colormap name, gray etc...

## CONTOUR PLOTS                                                           

## ERRORBAR PLOTS                                                          

## * HISTOGRAM PLOTS                                                         
hist.bins: auto  # The default number of histogram bins or 'auto'.

## SCATTER PLOTS                                                           

## AGG RENDERING                                                           

## PATHS                                                                   

## SAVING FIGURES                                                          
savefig.dpi:       300      # figure dots per inch or 'figure'
savefig.format:    png         # {png, ps, pdf, svg}
savefig.facecolor: white     # figure face color when saving
savefig.bbox:      tight    # {tight, standard}
                                # 'tight' is incompatible with pipe-based animation
                                # backends (e.g. 'ffmpeg') but will work with those
                                # based on temporary files (e.g. 'ffmpeg_file')
savefig.transparent: False     # setting that controls whether figures are saved with a
                                # transparent background by default

## INTERACTIVE KEYMAPS                                                     
## See https://matplotlib.org/users/navigation_toolbar.html 

## ANIMATION                                                               
"""


def make_matplotlib_rc(rcparams_str=RC_STR):
    """
    Make a matplotlib style sheet (.rc file) with defaults specified below.

    See https://matplotlib.org/stable/tutorials/introductory/customizing.html
    """
    # Set custom matplotlibrc parameters below

    dstpath = Path(get_configdir()).joinpath('matplotlibrc')
    if not dstpath.exists():
        print(f'Making matplotlib rc file at {dstpath}.')
        dstpath.parent.mkdir(parents=True, exist_ok=True)
        with open(dstpath, 'w') as f:
            f.write(rcparams_str)
        print('Make sure to restart all Python and Jupyter instances.')
    else:
        print(f'File {dstpath} already exists.')
        print('Move/delete file then run this again to make new matplotlibrc.')


def install_fonts(path_to_ttf=DOWNLOADS_DIR):
    """
    Install .ttf fonts in path_to_ttf to use in matplotlib. 

    E.g. to use the font Myriad Pro, download and unzip the font from:
        https://www.cufonfonts.com/font/myriad-pro

    Then call install_fonts("path/to/myriad-pro/")

    Adapted from L. Davis (https://stackoverflow.com/a/47743010/8742181)
    """    
    # Search for font ttf files to add to matplotlib
    font_paths = list(Path(path_to_ttf).rglob('*.[ot]tf'))
    if len(font_paths) == 0:
        if path_to_ttf != DOWNLOADS_DIR:
            # Give message if path specified and no fonts found
            print(f"No fonts found in {path_to_ttf}")
        return


    # Make matplotlib font directory
    dir_dst = Path(matplotlib_fname()).parent.joinpath('fonts', 'ttf')
    if not dir_dst.exists():
        dir_dst.mkdir(parents=True, exist_ok=True)
    
    # Add new fonts to directory, reset cache if any added
    new_font_added = False
    for fpath in font_paths:
        if not dir_dst.joinpath(fpath.name).exists():
            print(f'Adding font "{fpath.name}".')
            shutil.copy2(fpath, dir_dst)
            new_font_added = True
    
    if new_font_added:
        reset_matplotlib_cache()
    else:
        print(f'All fonts at {path_to_ttf} already added.')


def reset_matplotlib_cache():
    """Matplotlib cache must be deleted for new fonts to register"""
    print('Resetting matplotlib cache')
    dir_cache = get_cachedir()
    for re in ('*.cache', 'font*'):
        for path in Path(dir_cache).rglob(re):
            if 'tex' not in path.name:
                path.unlink()    
                print(f'Deleted cache {path.name}.')


if __name__ == '__main__':
    make_matplotlib_rc()
    if len(sys.argv) > 1:
        install_fonts(sys.argv[1])
    else:
        install_fonts()
