import argparse, logging, os, sys
import matplotlib.pyplot as plt, numpy as np
from src.pampelmuse.core import my_profiles

parser = argparse.ArgumentParser(fromfile_prefix_chars="@", description="""
Plot one or more PSF profiles using the analytical functions that are
implemented in PampelMuse.""")
parser.add_argument("--gauss",  nargs=3, action="append", help="""
Plot a Gaussian PSF profile. This option might be specified multiple times
to plot more than one profile. For each profile, the FWHM (in pixels) and
the style and color of the line used in the plot must be given.""")
parser.add_argument("--moffat",  nargs=4, action="append", help="""
Plot a Moffat PSF profile. This option might be specified multiple times
to plot more than one profile. For each profile, beta, the FWHM (in pixels)
and the style and color of the line used in the plot must be given.""")
## parser.add_argument("--sersic", nargs=4)
parser.add_argument("-r", "--radius", type=int, default=5, help="""
The radius (in pixels) out to which the profiles are plotted. Default is 5.""")
parser.add_argument("-s", "--sampling", type=float, default=1., help="""
The number of datapoints plotted per pixel. Default is 1.""")
parser.add_argument("--logy", action="store_true", default=False, help="""
Set this flag to plot with log10-scaling along the y-axis.""")
parser.add_argument("--save", type=str,  help="""
To save the resulting plot, provide a filename including the extension for
the requested format (e.g `eps`, `pdf` or `png`).""")

lineStyles = {"solid":"-", "dashed":"--", "dotted":":", "dotdashed":"-."}


args = parser.parse_args()

if args.moffat is None and args.gauss is None:
    logging.error("Nothing to plot here. Exiting...")
    sys.exit()

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_axes([0.10,0.08,0.86,0.88])
ax1.minorticks_on()

xData  = np.arange(start=0, stop=2*args.radius+1, step=1./args.sampling,
                   dtype=np.float32)
xData -= args.radius



def getLStyle(string):

    if not string in lineStyles.keys():
        logging.error("Unknown linestyle: `{0}`".format(string))
        logging.error("Linstyle must be in: [{0}]".format(\
                ", ".join(lineStyles.keys())))

        return None
    return lineStyles[string]
    

    
def plotProf(yData, ls, c, label):

    ax1.plot(xData[:-args.sampling+1],
             yData[0,:-args.sampling+1]*(args.sampling**2),
             ls=ls, c=c, marker="None", label=label)
             

if args.gauss is None:
    for i in xrange(len(args.gauss)):
        
        fwhm = float(args.gauss[i][0])*args.sampling
        ls   = getLStyle(args.gauss[i][1])
        c    = args.gauss[i][2]
        
        gauss_i = my_profiles.Gaussian(fwhm=fwhm)
        gaussProf_i = gauss_i(size=(1, xData.size),
                              x_c=args.radius*args.sampling, y_c=0, mag=0.)

        plotProf(gaussProf_i, ls=ls, c=c, 
                 label=r"Gaussian (${\rm FWHM}=%.1f$)" %(fwhm/args.sampling))


if args.moffat is None:
    for i in xrange(len(args.moffat)):
    
        beta = float(args.moffat[i][0])
        fwhm = float(args.moffat[i][1])*args.sampling
        ls   = getLStyle(args.moffat[i][2])
        c    = args.moffat[i][3]

        moffat_i = my_profiles.Moffat(beta=beta, fwhm=fwhm)
        moffatProf_i = moffat_i(size=(1, xData.size),
                                x_c=args.radius*args.sampling, y_c=0,
                                mag=0.)
        
        plotProf(moffatProf_i, ls=ls, c=c,
                 label=r"Moffat (${\rm FWHM}=%.1f,\, \beta=%.1f$)" \
                     %(fwhm/args.sampling, beta))



leg = ax1.legend()
leg.draw_frame(False)

if args.logy:
    ax1.set_yscale("log", basey=10)
    ax1.set_ylim(ax1.get_ylim()[0], 1)
else:
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax+0.2*(ymax-ymin))

ax1.set_xlabel(r"$r$ [${\rm pixel}$]", fontsize=18)
ax1.set_ylabel(r"intensity", fontsize=18)

if args.save is None:
    
    extension = args.save.split(".")[-1]

    if extension == "eps":
        fig.savefig(args.save[:-4]+".pdf")
        os.system("pdftops -paper match -eps {0}.pdf".format(args.save[:-4]))
        os.remove("{0}.pdf".format(args.save[:-4]))
    else:
        fig.savefig(args.save)

plt.show()
