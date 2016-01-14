import numpy as np
import matplotlib
matplotlib.use('qt4agg')
# 'pgf' no pop up windows, runs xelatex
# 'cairo' passes, no pop up
# 'MacOSX'
# 'CocoaAgg' fails, requires PyObjC
# 'gdk' runs and freezes, no pop up
# 'ps' passes, no pop up
# 'GTKAgg' first window opp up, then freezes
# 'nbAgg' fails
# 'GTK' first window pop upn, than fails: RuntimeError: could not create
# GdkPixmap object
# 'Qt5Agg' fails
# 'template' passes, but no pop up
# 'emf' fails, no backend emf
# 'GTK3Cairo' first window pop up then freeze
# 'GTK3Agg' first window pop up then freeze
# 'WX' first window pop up then fails
# 'Qt4Agg' first window pop up then freeze
# 'TkAgg' fails cannot import _tkagg
# 'agg' runs, but no pop up
# 'svg' runs, but no pop up
# 'GTKCairo' empty window pop up, error message, freeze
# 'WXAgg' first window pop up then fails
# 'WebAgg' opens window in webbrowser, then freezes
# 'pdf'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# fig = plt.figure()
fig = matplotlib.figure.Figure()

fig.suptitle('SUPER TITLE')

# cm = plt.cm.Greys
cm = matplotlib.cm.Greys

ax = fig.add_subplot(111)
ax.set_title('ARRAY')
ax.axis('off')
# print(ax)

im = ax.imshow(np.ones((100,100)), cmap=cm, interpolation='none')

# Get axes from image instance: matplotlib.axes._subplots.AxesSubplot
ax = im.axes

# Create divider for existing axes instance
divider = make_axes_locatable(ax)

# Append axes to the right of ax3, with 20% width of ax3
cax = divider.append_axes("right", size="5%", pad=0.04)

# Change colorbar ticks format
scalform = ticker.ScalarFormatter()
scalform.set_scientific(True)
scalform.set_powerlimits((-1, 1))

# Create colorbar in the appended axes
# cbar = plt.colorbar(image, cax=cax, ticks=ticker.MultipleLocator(0.2),
#                     format="%.2f")
cbar = plt.colorbar(im, cax=cax, format=scalform)
print(type(cbar))
cbar = matplotlib.figure.Figure.colorbar(fig, im, cax=cax, format=scalform)
print(type(cbar))

canvas = matplotlib.backend_bases.FigureCanvasBase(fig)
print(type(canvas))

fmb = matplotlib.backend_bases.FigureManagerBase(canvas, 0)
print(type(fmb))

fmb.set_window_title('WINDOW TITLE')
wt = fmb.get_window_title()
print(wt)