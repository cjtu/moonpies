from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from moonpies import moonpies as mp
from moonpies import default_config

# Set Fig paths
FIGDIR = ''  # Set or leave blank to use default (moonpies/figs)
if not FIGDIR:
    FIGDIR = default_config.Cfg().figs_path
FIGDIR = str(Path(FIGDIR).resolve() / "_")[:-1]  # add trailing slash

cfg = default_config.Cfg()

cold_trap_name_moores = ["Haworth", "Shoemaker", "Faustini", "de Gerlache", "Sverdrup", "Shackleton", "Cabeus"]
cold_trap_lat_moores = [87.5,88.,87.1,88.3,88.3,89.6,84.5]
cold_trap_name = ['Haworth', 'Shoemaker', 'Faustini', 'Amundsen','Cabeus','Cabeus B', 'de Gerlache', 
                  "Idel'son L", 'Sverdrup', 'Shackleton', 'Wiechert J', "Slater"]
cold_trap_lat = np.array([87.5, 88., 87.1, 84.4, 85.3, 82.3, 88.3, 84., 88.3, 89.6, 85.2, 88.1])
label_offset = np.array([(-0.7, 0.1), (-0.9, -0.8), (0.1, 0), (-0.1, -0.8), (0.1, -0.5), (0, 0.1), (0.1, 0.25), 
                        (-0.8, -0.8), (0, -1), (-0.95, -0.4), (-0.5, 0.25), (-0.45, 0.05)])

def plot_bhop():
    plt.figure(figsize=(8,4))
    bhops = mp.read_ballistic_hop_csv(cfg.bhop_csv_in)
    for i, (name, lat) in enumerate(zip(cold_trap_name, cold_trap_lat)):
        bhop = 100*bhops[name]
        color = 'tab:orange'
        marker = 's'
        label = None
        if name in cold_trap_name_moores:
            color = 'tab:blue'
            marker = 'o'
        if name == 'Haworth':
            label = 'Moores'
        elif name == 'Cabeus B':
            label = 'This work'
            
        plt.plot(lat, bhop, marker, c=color, label=label)
        off_x, off_y = label_offset[i]
        plt.annotate(name, (lat, bhop), xytext=(lat + off_x, bhop+off_y), ha='left', va='bottom')
    plt.axhline(5.2, c='tab:gray')
    plt.annotate('Cannon et al. (2020)', (82.1, 5.1), va='top')
    # plt.text(82, 100*(0.052 + 0.001), "Cannon et al. 2020")

    bhop_moores = [100*bhops[name] for name in cold_trap_name_moores]
    m, b = np.polyfit(cold_trap_lat_moores, bhop_moores, 1)
    lat = np.linspace(80, 90, 100)
    plt.plot(lat, (m*lat + b), '--')
    plt.annotate("Fit to Moores et al. (2016)", (83.5, 8.2))
    mp.plot_version(cfg, loc='ll')
    plt.legend()
    plt.xlim(82, 90)
    plt.ylim(0, 10)
    plt.xlabel("Latitude [Degrees]")
    plt.ylabel("Ballistic Hop Efficiency [% per km$^{2}$]")
    plt.title("Ballistic Hop Efficiency by Latitude")
    plt.tight_layout()
    plt.savefig(FIGDIR + "bhop_lat.pdf")
plot_bhop()
quit()

def Moores():
    names = ["Haworth", "Shoemaker", "Faustini", "de Gerlache", "Sverdrup", "Shackleton","Cabeus"]
    Nho = np.array([11617.,10190.,21515.,2851.,5512.,839.,16663.])
    lat = np.array([87.5,88.,87.1,88.3,88.3,89.6,84.5])
    diams = np.array([35.6, 39.9, 30.3, 20.3, 26.3, 20.1, 20.6])

    area = np.pi*(diams/2)**2
    rate = Nho/area

    frac = rate / 4800000.
    area_frac = Nho / 4800000.

    m,b = np.polyfit(lat, frac, 1)    
    return frac, area_frac, lat, names, m, b

def cold_trap():
    cold_trap_name = ['Haworth', 'Shoemaker', 'Faustini', 'Amundsen','Cabeus','Cabeus B', 'de Gerlache', "Idel'son L",  'Sverdrup', 'Shackleton', 'Wiechert J', "Slater"]
    cold_trap_lat = np.array([87.5, 88., 87.1, 84.4, 85.3, 82.3, 88.3, 84., 88.3, 89.6, 85.2, 88.1])
    cold_trap_area = np.array([1017.932,1075.518,663.934,701.959,0,387.205,243.292,326.779,548.791,371.549,233.698,0])
    return cold_trap_name, cold_trap_lat, cold_trap_area

def plot_km(lat, frac, m, b, cold_trap_lat, cold_trap_name, cfg):
    plt.figure(figsize=(6,4))
    plt.plot(lat[:-1], 1e6*frac[:-1], 'o', label="Moores 2016")
    plt.plot(cold_trap_lat[3], 1e6*(m*cold_trap_lat[3] + b), 'ro', label="This work")
    plt.plot(cold_trap_lat[4], 1e6*(m*cold_trap_lat[4] + b), 'ro')
    plt.plot(cold_trap_lat[5], 1e6*(m*cold_trap_lat[5] + b), 'ro')
    plt.plot(cold_trap_lat[7], 1e6*(m*cold_trap_lat[7] + b), 'ro')
    plt.plot(cold_trap_lat[-1], 1e6*(m*cold_trap_lat[-1] + b), 'ro')
    plt.plot(cold_trap_lat,1e6*(m*cold_trap_lat + b), '--', label="Fit to Moores 2016")

    plt.axhline(100*(0.056), label="Cannon et al. 2020")
    plt.text(82, 100*(0.056 + 0.001), "Cannon et al. 2020")
    plt.text(86.7, 100*(0.056 + 0.02), "Fit to Moores 2016")
    plt.arrow(86.7, 100*(0.056 + 0.02), -0.5, -1, width=0.0000001, head_width=0.00002, head_length=0.0001, shape='full')
#    for f in range(0,5):#len(cold_trap_lat)):#
#        plt.text(cold_trap_lat[f], 100*(m*cold_trap_lat[f] + b), cold_trap_name[f])
 
    plt.text(cold_trap_lat[0]-1.2, 1e6*(m*cold_trap_lat[0] + b)-1.2, cold_trap_name[0])
    plt.text(cold_trap_lat[1]-1.6, 1e6*(m*cold_trap_lat[1] + b)-1.8, cold_trap_name[1])
    plt.text(cold_trap_lat[2]+0.1, 1e6*(m*cold_trap_lat[2] + b)+1.5, cold_trap_name[2])
    plt.text(cold_trap_lat[3]+0.1, 1e6*(m*cold_trap_lat[3] + b), cold_trap_name[3])
    plt.text(cold_trap_lat[4]+0.1, 1e6*(m*cold_trap_lat[4] + b)-.25, cold_trap_name[4])
    plt.text(cold_trap_lat[5]+0.1, 1e6*(m*cold_trap_lat[5] + b), cold_trap_name[5])
    plt.text(cold_trap_lat[7]+0.1, 1e6*(m*cold_trap_lat[7] + b), cold_trap_name[7])
    plt.text(cold_trap_lat[-1]+0.1, 1e6*(m*cold_trap_lat[-1] + b)+0.2, cold_trap_name[-1])
    plt.text(cold_trap_lat[-2]+0.1, 1e6*(m*cold_trap_lat[-2] + b)+0.3, cold_trap_name[-2])
    plt.text(cold_trap_lat[-3]-1.45, 1e6*(m*cold_trap_lat[-3] + b)+0.005, cold_trap_name[-3])
    plt.text(cold_trap_lat[6]+0.1, 1e6*(m*cold_trap_lat[6] + b), cold_trap_name[6])
    plt.text(cold_trap_lat[8]+0.1, 1e6*(m*cold_trap_lat[8] + b)-0.7, cold_trap_name[8])
    #plt.text(lat[-1], 13000*100*(m*lat[-1]+b-0.0000003), "Cabeus")
    mp.plot_version(cfg, loc='ll')
    plt.legend()
    plt.xlabel("Latitude [Degrees]")
    plt.ylabel("Ballistic Hop Efficiency [% per km$^{2}$]")
    plt.title("Ballistic Hop Efficiency by Latitude")
    plt.tight_layout()
    plt.savefig(FIGDIR + "ball_hop_km.png", dpi=300)
    # plt.show()

def plot_area(m, b, cold_trap_lat, cold_trap_name, cold_trap_area, cfg):
    for f in range(0,len(cold_trap_area)-1):
        plt.plot(cold_trap_lat[f], cold_trap_area[f]*100*(m*cold_trap_lat[f] + b), 'ro')#, label="This work")
    plt.plot(cold_trap_lat[-1], cold_trap_area[-1]*100*(m*cold_trap_lat[-1] + b), 'ro')#

    for f in range(0,5):#len(cold_trap_lat)):#
        plt.text(cold_trap_lat[f], cold_trap_area[f]*100*(m*cold_trap_lat[f] + b), cold_trap_name[f])#
    plt.text(cold_trap_lat[6], cold_trap_area[6]*100*(m*cold_trap_lat[6] + b), cold_trap_name[6])#
    plt.text(cold_trap_lat[-1], cold_trap_area[-1]*100*(m*cold_trap_lat[-1] + b), cold_trap_name[-1])#
    plt.text(cold_trap_lat[-2], cold_trap_area[-2]*100*(m*cold_trap_lat[-2] + b), cold_trap_name[-2])#
    plt.text(cold_trap_lat[5]+0.1, cold_trap_area[5]*100*(m*cold_trap_lat[5] + b), cold_trap_name[5]+",")#
    plt.text(cold_trap_lat[7]+0.1, cold_trap_area[7]*100*((m*cold_trap_lat[7] + b)-0.0000005), cold_trap_name[7])#
    #plt.text(lat[-1], 13000*100*(m*lat[-1]+b-0.0000003), "Cabeus")
    plt.xlabel("Latitude [degrees]")
    plt.ylabel("Ballistic hop efficiency [% per total PSR area]")
    plt.title("Ballistic hop efficiency by latitude")
    plt.tight_layout()
    plt.savefig(FIGDIR + "ball_hop_area.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    cfg = default_config.Cfg(mode='moonpies')

    frac, area_frac, lat, names, m, b = Moores()
    cold_trap_name, cold_trap_lat, cold_trap_area = cold_trap()
    plot_km(lat, frac, m, b, cold_trap_lat, cold_trap_name, cfg)
#plot_area(m, b, cold_trap_lat, cold_trap_name, cold_trap_area, cfg)
#cold_trap_fit = m*cold_trap_lat + b
#np.savetxt("ball_hop_fit.csv", cold_trap_fit)
#np.savetxt("ball_hop_moores.csv", frac)

#df_moores = pd.DataFrame(frac, index=[names])
#df_fit = pd.DataFrame(cold_trap_fit, index=[cold_trap_name])

#df_moores.to_csv("moores_ball_hop.csv")
#df_fit.to_csv("fit_ball_hop.csv")