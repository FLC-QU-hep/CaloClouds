
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

mpl.rcParams['xtick.labelsize'] = 25    
mpl.rcParams['ytick.labelsize'] = 25
# mpl.rcParams['font.size'] = 28
mpl.rcParams['font.size'] = 35
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'



class Configs():
    
    def __init__(self):

    # legend font
        self.font = font_manager.FontProperties(
            family='serif',
            size=23
            # size=20
        )
        self.text_size = 20
        
    # radial profile
        self.bins_r = 35
        self.origin = (0, 40)
        # self.origin = (3.754597092*10, -3.611833191*10)
        

    # occupancy
        self.occup_bins = np.linspace(150, 1479, 100)
        self.plot_text_occupancy = False
        self.occ_indent = 20

    # e_sum
        self.e_sum_bins = np.linspace(20.01, 2400, 150)
        self.plot_text_e = False
        self.plot_legend_e = True
        self.e_indent = 20

    # hits
        self.hit_bins = np.logspace(np.log10(0.01000001), np.log10(1000), 70)
        # self.hit_bins = np.logspace(np.log10(0.01), np.log10(100), 70)
        self.ylim_hits = (10, 3*1e7)
        # self.ylim_hits = (10, 8*1e5)

    #CoG
        self.bins_cog = 30  
        # bin ranges for [X, Z, Y] coordinates, in ILD coordinate system [X', Y', Z']
        self.cog_ranges = [(-3.99+1.5, 3.99+1.5), (1861, 1999), (36.01+1.5, 43.99+1.5)]
        # self.cog_ranges = [(-3.99, 3.99), (1861, 1999), (36.01, 43.99)]
        # self.cog_ranges = [(33.99, 39.99), (1861, 1999), (-38.9, -32.9)]


    # all
        # self.color_lines = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        # self.color_lines = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
        self.color_lines = ['tab:orange', 'tab:orange', 'tab:orange', 'tab:orange']

        self.include_artifacts = True


cfg = Configs()

Ymin = 1811
Xmin = -200
Xmax = 200
# Xmin = -260
# Xmax = 340

Zmin = -160
Zmax = 240
# Zmin = -300
# Zmax = 300

half_cell_size = 5.0883331298828125/2
cell_thickness = 0.5250244140625

layer_bottom_pos = np.array([   1811.34020996, 1814.46508789, 1823.81005859, 1826.93505859,
                                    1836.2800293 , 1839.4050293 , 1848.75      , 1851.875     ,
                                    1861.2199707 , 1864.3449707 , 1873.68994141, 1876.81494141,
                                    1886.16003418, 1889.28503418, 1898.63000488, 1901.75500488,
                                    1911.09997559, 1914.22497559, 1923.56994629, 1926.69494629,
                                    1938.14001465, 1943.36499023, 1954.81005859, 1960.03503418,
                                    1971.47998047, 1976.70495605, 1988.15002441, 1993.375     ,
                                    2004.81994629, 2010.04504395])

X = np.load('./data/calo-clouds/muon-map/X.npy')
Z = np.load('./data/calo-clouds/muon-map/Z.npy')
Y = np.load('./data/calo-clouds/muon-map/Y.npy')
E = np.load('./data/calo-clouds/muon-map/E.npy')

inbox_idx = np.where((Y > Ymin) & (X < Xmax) & (X > Xmin) & (Z < Zmax) & (Z > Zmin) )[0]


X = X[inbox_idx]
Z = Z[inbox_idx]
Y = Y[inbox_idx]
E = E[inbox_idx]


def create_map(X, Y, Z, dm=3):
    """
        X, Y, Z: np.array 
            ILD coordinates of sencors hited with muons
        dm: int (1, 2, 3, 4, 5) dimension split multiplicity
    """

    offset = half_cell_size*2/(dm)

    layers = []
    for l in tqdm(range(len(layer_bottom_pos))): # loop over layers
        idx = np.where((Y <= (layer_bottom_pos[l] + cell_thickness*1.5)) & (Y >= layer_bottom_pos[l] - cell_thickness/2 ))
        
        xedges = np.array([])
        zedges = np.array([])
        
        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])
        
        xedges = np.append(xedges, unique_X[0] - half_cell_size)
        xedges = np.append(xedges, unique_X[0] + half_cell_size)
        
        for i in range(len(unique_X)-1): # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i+1]) > half_cell_size * 1.9:
                xedges = np.append(xedges, unique_X[i+1] - half_cell_size)
                xedges = np.append(xedges, unique_X[i+1] + half_cell_size)
                
                for of_m in range(dm):
                    xedges = np.append(xedges, unique_X[i+1] - half_cell_size + offset*of_m) # for higher granularity
                
        for z in unique_Z: # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size)
            zedges = np.append(zedges, z + half_cell_size)
            
            for of_m in range(dm):
                zedges = np.append(zedges, z - half_cell_size + offset*of_m) # for higher granularity
                
            
        zedges = np.unique(zedges)
        xedges = np.unique(xedges)
        
        xedges = [xedges[i] for i in range(len(xedges)-1) if abs(xedges[i] - xedges[i+1]) > 1e-3] + [xedges[-1]]
        zedges = [zedges[i] for i in range(len(zedges)-1) if abs(zedges[i] - zedges[i+1]) > 1e-3] + [zedges[-1]]
        
            
        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({'xedges': xedges, 'zedges': zedges, 'grid': H})

    return layers, offset




def get_projections(showers, MAP, layer_bottom_pos, return_cell_point_cloud=False):
    events = []
    
    for shower in tqdm(showers):
        layers = []
        
        x_coord, y_coord, z_coord, e_coord = shower

        
        for l in range(len(MAP)):
            idx = np.where((y_coord <= (layer_bottom_pos[l] + 1)) & (y_coord >= layer_bottom_pos[l] - 0.5 ))
            
            xedges = MAP[l]['xedges']
            zedges = MAP[l]['zedges']
            H_base = MAP[l]['grid']
            
            H, xedges, zedges = np.histogram2d(x_coord[idx], z_coord[idx], bins=(xedges, zedges), weights=e_coord[idx])
            if not cfg.include_artifacts:
                H[H_base==0] = 0
            
            layers.append(H)
        
        events.append(layers)
    
    if not return_cell_point_cloud:
        return events
    
    else:
        pass




def get_cog(x, y, z, e):
    x_cog = np.sum((x * e), axis=1) / e.sum(axis=1)
    y_cog = np.sum((y * e), axis=1) / e.sum(axis=1)
    z_cog = np.sum((z * e), axis=1) / e.sum(axis=1)
    return x_cog, y_cog, z_cog

def get_features(events, thr=0.05):
    
    incident_point = cfg.origin
    
    occ_list = [] # occupancy
    hits_list = [] # energy per cell
    e_sum_list = [] # energy per shower
    e_radial = [] # radial profile
    e_layers_list = [] # energy per layer

    for layers in tqdm(events):

        occ = 0
        e_sum = 0
        e_layers = []
        y_pos = []
        for l, layer in enumerate(layers):
            layer = layer*1000 # energy rescale
            layer[layer < thr] = 0

            hit_mask = layer > 0
            layer_hits = layer[hit_mask]
            layer_sum = layer.sum()

            occ += hit_mask.sum()
            e_sum += layer.sum()

            hits_list.append(layer_hits)
            e_layers.append(layer.sum())


            # get radial profile #######################
            x_hit_idx, z_hit_idx = np.where(hit_mask)
            x_cell_coord = MAP[l]['xedges'][:-1][x_hit_idx] + half_cell_size
            z_cell_coord = MAP[l]['zedges'][:-1][z_hit_idx] + half_cell_size
            e_cell = layer[x_hit_idx, z_hit_idx]
            dist_to_origin = np.sqrt((x_cell_coord - incident_point[0])**2 + (z_cell_coord - incident_point[1])**2)
            e_radial.append([dist_to_origin, e_cell])
            ############################################


        e_layers_list.append(e_layers)

        occ_list.append(occ)
        e_sum_list.append(e_sum)

    e_radial = np.concatenate(e_radial, axis=1)
    occ_list = np.array(occ_list)
    e_sum_list = np.array(e_sum_list)
    hits_list = np.concatenate(hits_list)
    e_layers_list = np.array(e_layers_list).sum(axis=0)/len(events)
    
    return e_radial, occ_list, e_sum_list, hits_list, e_layers_list


def plt_radial(e_radial, e_radial_list, labels, cfg=cfg):
    fig = plt.figure(figsize=(7,7))

    ## for legend ##########################################
    plt.hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    plt.plot(0, 0, linestyle='-', lw=3, color='tab:orange', label=r'\textsc{CaloClouds}')
    plt.title(r'\textbf{full spectrum}', fontsize=cfg.font.get_size(), loc='right')
    plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    ########################################################

    h = plt.hist(e_radial[0], bins=cfg.bins_r, weights=e_radial[1], color='lightgrey', rasterized=True)
    h = plt.hist(e_radial[0], bins=cfg.bins_r, weights=e_radial[1], color='dimgrey', histtype='step', lw=2)
    
    for i, e_radial_ in enumerate(e_radial_list):
        h = plt.hist(e_radial_[0], bins=h[1], weights=e_radial_[1], histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i])
        
    
    
    plt.yscale('log')
    
    plt.xlabel("radius [mm]")
    plt.ylabel('energy sum [MeV]')

    plt.tight_layout()

    plt.savefig('radial.pdf', dpi=100)
    plt.show()
    
def plt_spinal(e_layers, e_layers_list, labels, cfg=cfg):
    
    plt.figure(figsize=(7,7))

    ## for legend ##########################################
    plt.hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    plt.plot(0, 0, linestyle='-', lw=3, color='tab:orange', label=r'\textsc{CaloClouds}')
    ########################################################

    plt.hist(np.arange(len(e_layers)), bins=30, weights=e_layers, color='lightgrey', rasterized=True)
    plt.hist(np.arange(len(e_layers)), bins=30, weights=e_layers, color='dimgrey', histtype='step', lw=2)
    
    for i, e_layers_ in enumerate(e_layers_list):
        plt.hist(np.arange(len(e_layers_)), bins=30, weights=e_layers_, histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i])

    plt.yscale('log')
    plt.ylim(1, 1000)
    plt.xlabel('layers')
    plt.ylabel('energy sum [MeV]')
    
    plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    plt.title(r'\textbf{full spectrum}', fontsize=cfg.font.get_size(), loc='right')
    plt.tight_layout()

    plt.savefig('spinal.pdf', dpi=100)
    plt.show()
    
def plt_occupancy(occ, occ_list, labels, cfg=cfg):
    plt.figure(figsize=(7,7))

    ## for legend ##########################################
    plt.hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    plt.plot(0, 0, linestyle='-', lw=3, color='tab:orange', label=r'\textsc{CaloClouds}')
    ########################################################

    h = plt.hist(occ, bins=cfg.occup_bins, color='lightgrey', rasterized=True)
    h = plt.hist(occ, bins=cfg.occup_bins, color='dimgrey', histtype='step', lw=2)
    
    for i, occ_ in enumerate(occ_list):
        plt.hist(occ_, bins=h[1], histtype='step', linestyle='-', lw=2.5, color=cfg.color_lines[i])

    plt.xlim(cfg.occup_bins.min() - cfg.occ_indent, cfg.occup_bins.max() + cfg.occ_indent)
    plt.xlabel('number of hits')
    plt.ylabel('\# showers')

    plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    if cfg.plot_text_occupancy:
        plt.text(315, 540, '10 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(870, 215, '50 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(1230, 170, '90 GeV', fontsize=cfg.font.get_size() + 2)


    plt.tight_layout()
    plt.savefig('occ.pdf', dpi=100)
    plt.show()
    
def plt_hit_e(hits, hits_list, labels, cfg=cfg):
    plt.figure(figsize=(7,7))

    ## for legend ##########################################
    plt.hist(np.zeros(1)+1, label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    plt.plot(0, 0, linestyle='-', lw=3, color='tab:orange', label=r'\textsc{CaloClouds}')
    # plt.legend(prop=cfg.font, loc='upper right')
    plt.legend(prop=cfg.font, loc=(0.35, 0.78))
    # plt.title(r'\textbf{validation set, 50 GeV}', fontsize=cfg.font.get_size(), loc='right')
    plt.title(r'\textbf{full spectrum}', fontsize=cfg.font.get_size(), loc='right')
    ########################################################

    h = plt.hist(hits, bins=cfg.hit_bins, color='lightgrey', rasterized=True)
    h = plt.hist(hits, bins=cfg.hit_bins, histtype='step', color='dimgrey', lw=2)
    
    for i, hits_ in enumerate(hits_list):
        plt.hist(hits_, bins=h[1], histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i])

    plt.axvspan(h[1].min(), 0.1, facecolor='gray', alpha=0.5, hatch= "/", edgecolor='k')
    plt.xlim(h[1].min(), h[1].max()+0)
    plt.ylim(cfg.ylim_hits[0], cfg.ylim_hits[1])

    plt.yscale('log')
    plt.xscale('log')

    plt.xlabel('visible cell energy [MeV]')
    plt.ylabel('\# cells')


    plt.tight_layout()
    plt.savefig('hits.pdf', dpi=100)
    plt.show()
    
def plt_esum(e_sum, e_sum_list, labels, cfg=cfg):
    plt.figure(figsize=(7, 7))
    
    h = plt.hist(np.array(e_sum), bins=cfg.e_sum_bins, color='lightgrey', rasterized=True)
    h = plt.hist(np.array(e_sum), bins=cfg.e_sum_bins,  histtype='step', color='dimgrey', lw=2)

    ## for legend ##########################################
    plt.hist(np.zeros(10), label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
    plt.plot(0, 0, linestyle='-', lw=3, color='tab:orange', label=r'\textsc{CaloClouds}')
    ########################################################

    
    for i, e_sum_ in enumerate(e_sum_list):
        plt.hist(np.array(e_sum_), bins=h[1], histtype='step', linestyle='-', lw=2.5, color=cfg.color_lines[i])

    plt.xlim(cfg.e_sum_bins.min() - cfg.e_indent, cfg.e_sum_bins.max() + cfg.e_indent)
    plt.xlabel('energy sum [MeV]')
    plt.ylabel('\# showers')
    

    if cfg.plot_text_e:
        plt.text(300, 740, '10 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(1170, 250, '50 GeV', fontsize=cfg.font.get_size() + 2)
        plt.text(1930, 160, '90 GeV', fontsize=cfg.font.get_size() + 2)
        plt.ylim(0, 799)

    if cfg.plot_legend_e:
        plt.legend(prop=cfg.font, loc=(0.35, 0.78))

    plt.tight_layout()
    plt.savefig('e_sum.pdf', dpi=100)
    plt.show()

def plt_cog(cog, cog_list, labels, cfg=cfg):
    lables = ["X", "Z", "Y"] # local coordinate system
    plt.figure(figsize=(21, 7))

    for k, j in enumerate([0, 2, 1]):
        plt.subplot(1, 3, k+1)

        plt.xlim(cfg.cog_ranges[j])

        
        h = plt.hist(np.array(cog[j]), bins=cfg.bins_cog, color='lightgrey', range=cfg.cog_ranges[j], rasterized=True)
        h = plt.hist(np.array(cog[j]), bins=h[1], color='dimgrey', histtype='step', lw=2)
        
        # for legend ##############################################
        if k == k:
        #     plt.plot(0, 0, lw=2, color='black', label=labels[0])
            plt.hist(np.zeros(10), label=labels[0], color='lightgrey', edgecolor='dimgrey', lw=2)
            plt.plot(0, 0, linestyle='-', lw=3, color='tab:orange', label=r'\textsc{CaloClouds}')
        ###########################################################

        for i, cog_ in enumerate(cog_list):
            h2 = plt.hist(np.array(cog_[j]), bins=h[1], histtype='step', linestyle='-', lw=3, color=cfg.color_lines[i], range=cfg.cog_ranges[j])

        # for legend ##############################################
        if k == k:
            plt.legend(prop=cfg.font, loc=(0.37, 0.76))

        ax = plt.gca()
        plt.title(r'\textbf{full spectrum}', fontsize=cfg.font.get_size(), loc='right')

        ###########################################################


        plt.ylim(0, max(h[0]) + max(h[0])*0.1)

        plt.xlabel(f'center of gravity {lables[j]} [mm]')
        plt.ylabel('\# showers')

    
    plt.tight_layout()
    plt.savefig('cog.pdf', dpi=100)
    plt.show()



def get_plots(events, events_list: list, labels: list = ['1', '2', '3'], thr=0.05):
    
    e_radial_real, occ_real, e_sum_real, hits_real, e_layers_real = get_features(events, thr)
    
    e_radial_list, occ_list, e_sum_list, hits_list, e_layers_list = [], [], [], [], []
    
    for i in range(len(events_list)):
        e_radial_, occ_real_, e_sum_real_, hits_real_, e_layers_real_ = get_features(events_list[i], thr)
        
        e_radial_list.append(e_radial_)
        occ_list.append(occ_real_)
        e_sum_list.append(e_sum_real_)
        hits_list.append(hits_real_)
        e_layers_list.append(e_layers_real_)
        
    
    plt_radial(e_radial_real, e_radial_list, labels=labels)
    plt_spinal(e_layers_real, e_layers_list, labels=labels)
    plt_hit_e(hits_real, hits_list, labels=labels)
    # plt_occupancy(occ_real, occ_list, labels=labels)
    # plt_esum(e_sum_real, e_sum_list, labels=labels)


MAP, offset = create_map(X, Y, Z, dm=1)
