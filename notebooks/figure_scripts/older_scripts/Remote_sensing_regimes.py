#plots depth of remote sensing regimes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections


x = np.linspace(0., 5., 50)
d = np.zeros(50)

fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10) = plt.subplots(1,10,sharey='all')#, figsize=[5,5])
fig.subplots_adjust(hspace=0.3, wspace=0.0)
ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
ax6.set_xticks([])
ax7.set_xticks([])
ax8.set_xticks([])
ax9.set_xticks([])
ax10.set_xticks([])
#ax2.set_yticks([],[])
#ax3.set_yticks([],[])

ax1.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x,ymin = 0.0, ymax = 10**-1, where=x, facecolor="paleturquoise")
ax1.add_collection(collection2)
ax1.set_xlabel("VIS-NIR", rotation=45)
ax1.set_ylim(10**-2,10**4)
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_xlim(0.1,4.9)
ax1.set_ylabel("Depth (m)")

ax2.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x,ymin = 0.0, ymax = 10**-1, where=x, facecolor="paleturquoise")
ax2.add_collection(collection2)
ax2.set_xlabel("UV", rotation=45)
ax2.set_ylim(10**-2,10**4)
ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.set_xlim(0.1,4.9)

ax3.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x,ymin = 0.0, ymax = 10**0, where=x, facecolor="paleturquoise")
ax3.add_collection(collection2)
ax3.set_xlabel("Neutron", rotation=45)
ax3.set_ylim(10**-2,10**4)
ax3.set_ylim(ax3.get_ylim()[::-1])
ax3.set_xlim(0.1,4.9)
ax3.set_title('Orbital Instruments')
#ax3.spines[axis].set_linewidth(2)
#ax3.axvline(4.9,linewidth=3,color='w')

ax4.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.1, ymax = 10**0, where=x, facecolor="paleturquoise")
ax4.add_collection(collection2)
ax4.set_xlabel("Radar", rotation=45)
ax4.set_ylim(10**-2,10**4)
ax4.set_ylim(ax4.get_ylim()[::-1])
ax4.set_xlim(0.1,4.9)

#ax4.axvline(0.1,linewidth=3,color='w')

ax5.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**2, ymax = 10**4, where=x, facecolor="paleturquoise")
ax5.add_collection(collection1)
#collection2 = collections.BrokenBarHCollection.span_where(
#  x, ymin = 10**1, ymax = 10**3, where=x, facecolor="steelblue")
#ax8.add_collection(collection2)
ax5.set_xlabel("Gravimeter", rotation=45)
ax5.set_ylim(10**-2,10**4)
ax5.set_ylim(ax5.get_ylim()[::-1])
ax5.set_xlim(0.1,4.9)

ax6.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.0, ymax = 10**0, where=x, facecolor="paleturquoise")
ax6.add_collection(collection1)
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**0, ymax = 10**.45, where=x, facecolor="steelblue")
#ax2.add_collection(collection2)
ax6.set_xlabel("VIPER Drill", rotation=45)
ax6.set_ylim(10**-2,10**4)
ax6.set_ylim(ax6.get_ylim()[::-1])
ax6.set_xlim(0.1,4.9)
#ax6.set_ylabel("Depth (m)")

ax6.axvline(0.1,linewidth=3,color='k')

ax7.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.0, ymax = 10**.45, where=x, facecolor="paleturquoise")
ax7.add_collection(collection1)
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**0, ymax = 10**.45, where=x, facecolor="steelblue")
#ax3.add_collection(collection2)
ax7.set_xlabel("Apollo Drill", rotation=45)
ax7.set_ylim(10**-2,10**4)
ax7.set_ylim(ax7.get_ylim()[::-1])
ax7.set_xlim(0.1,4.9)

#ax7.axvline(0.1,linewidth=3,color='w')

ax8.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**-1, ymax = 10**1, where=x, facecolor="paleturquoise")
ax8.add_collection(collection2)
ax8.set_xlabel("Impactor", rotation=45)
ax8.set_ylim(10**-2,10**4)
ax8.set_ylim(ax8.get_ylim()[::-1])
ax8.set_xlim(0.1,4.9)
ax8.set_title('Ground Instruments')

#ax8.axvline(0.1,linewidth=3,color='w')

ax9.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.0, ymax = 10**1, where=x, facecolor="paleturquoise")
ax9.add_collection(collection1)
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**1, ymax = 10**3, where=x, facecolor="steelblue")
ax9.add_collection(collection2)
ax9.set_xlabel("GPR", rotation=45)
ax9.set_ylim(10**-2,10**4)
ax9.set_ylim(ax9.get_ylim()[::-1])
ax9.set_xlim(0.1,4.9)

#ax9.axvline(0.1,linewidth=3,color='w')

ax10.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**2, ymax = 10**4, where=x, facecolor="paleturquoise")
ax10.add_collection(collection1)
#collection2 = collections.BrokenBarHCollection.span_where(
#  x, ymin = 10**1, ymax = 10**3, where=x, facecolor="steelblue")
#ax9.add_collection(collection2)
ax10.set_xlabel("Seismometer", rotation=45)
ax10.set_ylim(10**-2,10**4)
ax10.set_ylim(ax10.get_ylim()[::-1])
ax10.set_xlim(0.1,4.9)

fig.suptitle("Instrument Detection Regimes")
plt.savefig("regimes.png", dpi=300, bbox_inches = "tight")



fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5,sharey='all')#, figsize=[5,5])
fig.subplots_adjust(hspace=0.1, wspace=0.0)

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])
ax4.set_xticks([])
ax5.set_xticks([])
#ax6.set_xticks([],[])
#ax2.set_yticks([],[])
#ax3.set_yticks([],[])

ax1.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x,ymin = 0.0, ymax = 10**0, where=x, facecolor="paleturquoise")
ax1.add_collection(collection2)
ax1.set_xlabel("Neutron")
ax1.set_ylabel('Detection Depth [m]')
ax1.set_ylim(10**-2,10**4)
ax1.set_ylim(ax1.get_ylim()[::-1])
ax1.set_xlim(0,4.9)

#ax3.spines[axis].set_linewidth(2)
ax1.axvline(4.9,linewidth=3,color='w')

ax2.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.0, ymax = 10**0, where=x, facecolor="paleturquoise")
ax2.add_collection(collection1)
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**0, ymax = 10**.45, where=x, facecolor="steelblue")
#ax2.add_collection(collection2)
ax2.set_xlabel("VIPER Drill")
ax2.set_ylim(10**-2,10**4)
ax2.set_ylim(ax2.get_ylim()[::-1])
ax2.set_xlim(0.1,4.9)

ax2.axvline(0.1,linewidth=3,color='w')

ax3.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.0, ymax = 10**.45, where=x, facecolor="paleturquoise")
ax3.add_collection(collection1)
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**0, ymax = 10**.45, where=x, facecolor="steelblue")
#ax3.add_collection(collection2)
ax3.set_xlabel("Apollo Drill")
ax3.set_ylim(10**-2,10**4)
ax3.set_ylim(ax3.get_ylim()[::-1])
ax3.set_xlim(0.1,4.9)

ax3.axvline(0.1,linewidth=3,color='w')

ax4.semilogy(x,d,'k-')
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**-1, ymax = 10**1, where=x, facecolor="paleturquoise")
ax4.add_collection(collection2)
ax4.set_xlabel("Impactor")
ax4.set_ylim(10**-2,10**4)
ax4.set_ylim(ax4.get_ylim()[::-1])
ax4.set_xlim(0.1,4.9)

ax4.axvline(0.1,linewidth=3,color='w')

ax5.semilogy(x,d,'k-')
collection1 = collections.BrokenBarHCollection.span_where(
  x, ymin = 0.0, ymax = 10**0, where=x, facecolor="paleturquoise")
ax5.add_collection(collection1)
collection2 = collections.BrokenBarHCollection.span_where(
  x, ymin = 10**0, ymax = 10**3, where=x, facecolor="steelblue")
ax5.add_collection(collection2)
ax5.set_xlabel("GPR")
ax5.set_ylim(10**-2,10**4)
ax5.set_ylim(ax5.get_ylim()[::-1])
ax5.set_xlim(0.1,4.9)

ax5.axvline(0.1,linewidth=3,color='w')

#ax6.semilogy(x,d,'k-')
#collection1 = collections.BrokenBarHCollection.span_where(
#  x, ymin = 10**2, ymax = 10**4, where=x, facecolor="paleturquoise")
#ax6.add_collection(collection1)
##collection2 = collections.BrokenBarHCollection.span_where(
##  x, ymin = 10**1, ymax = 10**3, where=x, facecolor="steelblue")
##ax8.add_collection(collection2)
#ax6.set_xlabel("Gravimeter")
#ax6.set_ylim(10**-2,10**4)
#ax6.set_ylim(ax6.get_ylim()[::-1])
#ax6.set_xlim(0.1,4.9)

#ax9.semilogy(x,d,'k-')
#collection1 = collections.BrokenBarHCollection.span_where(
#  x, ymin = 10**2, ymax = 10**4, where=x, facecolor="paleturquoise")
#ax9.add_collection(collection1)
##collection2 = collections.BrokenBarHCollection.span_where(
##  x, ymin = 10**1, ymax = 10**3, where=x, facecolor="steelblue")
##ax9.add_collection(collection2)
#ax9.set_xlabel("Seismometer")
#ax9.set_ylim(10**-2,10**4)
#ax9.set_ylim(ax6.get_ylim()[::-1])
#ax9.set_xlim(0.1,4.9)

fig.suptitle("Instrument Detection Regimes")
plt.savefig("regimes_ground.jpg")