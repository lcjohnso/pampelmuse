import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib.path import Path

parser = argparse.ArgumentParser(description="""
Create a plot showing the arrangement of the spaxels in the field of view.""")
parser.add_argument("instrument", choices=["pmas", "argus"], help="""
The instrument for which the field of view should be plotted.""")

args = parser.parse_args()

if args.instrument == "pmas":
    instrument = Pmas()
elif args.instrument == "argus":
    instrument = Argus()
else:
    raise IOError('Unknown instrument specifier: "{0}"'.format(args.instrument))

xdim = (instrument.x.min()-0.6, instrument.x.max()+0.6)
ydim = (instrument.y.min()-0.6, instrument.y.max()+0.6)

fig = plt.figure(figsize=(7.*(xdim[1]-xdim[0])/(ydim[1]-ydim[0]), 7))
ax = fig.add_axes([.01, .01, .98, .98], frameon=False)

ax.set_xlim(xdim)
ax.set_ylim(ydim)

codes = [Path.MOVETO,
         Path.LINETO,
         Path.LINETO,
         Path.LINETO,
         Path.CLOSEPOLY,
         ]

color = colors.ColorConverter()

j = 0
for i in xrange(instrument.nfiber):

    if j in instrument.bad:
        j += 1
        continue

    if i in instrument.aux_fibers:
        continue

    x = instrument.x[j]
    y = instrument.y[j]

    xmin = x-0.5
    xmax = x+0.5

    ymin = y-0.5
    ymax = y+0.5

    verts = [(xmin, ymin),  # left, bottom
             (xmin, ymax),  # left, top
             (xmax, ymax),  # right, top
             (xmax, ymin),  # right, bottom
             (xmin, ymin),]  # ignored

    path = Path(verts, codes)

    # print color.to_rgba("orange",alpha=float(i)/instrument.nfiber)

    patch_edge = patches.PathPatch(path, facecolor="None", lw=2)
    patch_face = patches.PathPatch(path, facecolor="orange", alpha=float(i)/instrument.nfiber, lw=0)
    ax.add_patch(patch_face)
    ax.add_patch(patch_edge)

    ax.text(x, y, i, transform=ax.transData, fontsize=12, va="center",
            ha="center")
    j += 1

ax.set_xticks([])
ax.set_yticks([])

fig.savefig("{0}.png".format(instrument.name))

plt.show()
