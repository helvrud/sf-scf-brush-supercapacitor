import os
import numpy as np
os.system("mkdir -p /run/user/$(id -u)/flow/output")
from cap import Cap
import matplotlib.pyplot as plt
default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
from IPython.display import display, Latex
import pprint
import contextlib
import io
from itertools import cycle
import veusz.embed as veusz
from multiprocessing import Pool, cpu_count
import socket
paramsHP = {
    'phi': 0.0,
    'N': 200,
    'cna': 0.4,
    'D': 500,
    'timeout': 600,
    'alpha': 0.3,
    'sigma':3.5e-10,
    # "cover_thickness": 0, 
    # "cover_epsilon": 4, 
    # "chi_subano": 5.,

}
def runBrush(cap):
    # print(f'alpha={alpha}')
    global iguess, use_iguess
    
    if iguess:
        print (iguess)
        cap.iguess_in = iguess
        cap.__str__()
    cap.loadData(silent=True)
    if cap.solved:
        # Brush.getTheta()
        # Brush.getProfiles()
        print(f"{cap.fname}: ready\n")
        if use_iguess:
            iguess = cap.fnameiguess
    else:
        print(f"{cap.fname}: failed\n")
    
    return cap
def Brushes(CAPS):
    global iguess, use_iguess
    iguess = None
    use_iguess = True
    caps = map(runBrush, CAPS)
    CAPSsolved = []
    for c in caps:
        if c.solved:
            CAPSsolved.append(c)
    return CAPSsolved

def Brushes_mp(CAPS):
    global iguess, use_iguess
    iguess = None
    use_iguess = False
    with Pool(processes=cpu_count()) as pool:  # adjust workers
        caps = pool.map(runBrush, CAPS)
    CAPSsolved = []
    for c in caps:
        if c.solved:
            CAPSsolved.append(c)
    return CAPSsolved




def getCAPS(pKs, cnas, chis, PHIs, Alphas, mp= False):
    global paramsHP
    CAPSdict = {}
    # key = ''
    for pK in pKs :

        for cna in cnas:
            
            for chi in chis:
                
                for phi in PHIs:
                    # key = f'pK{pK}'*bool(len(pKs)-1)
                    # key += f'cna{cna}'*bool(len(cnas)-1)    
                    # key += f'chi{chi}'*bool(len(chis)-1)
                    # key += f'phi{phi}'*bool(len(PHIs)-1)
                    paramsHP.update({
                        'pK': pK,
                        'cna': cna,
                        'chi': chi,
                        'phi': phi
                        })
                    key = f'pK{pK}cna{cna}chi{chi}phi{phi}'
                    
                    
                    CAPS = []
                    for alpha in Alphas:
                        paramsHP.update({'alpha': alpha})    
                        # pprint.pprint(paramsHP)
                        Brush = Cap(**paramsHP)
                        # print(Brush.fname)
                        CAPS.append(Brush)
                    if mp:
                        CAPSSolved = Brushes_mp(CAPS)
                    else:
                        CAPSSolved = Brushes(CAPS)

                    for c in CAPSSolved: 
                        c.getTheta()
                        c.getProfiles()
                    CAPSdict[key] = CAPSSolved
                    


    return CAPSdict

def plotQV(
    CAPS,
    label='',
    ax=None,
    color=None,
    fill=False,
    title=None,
    xlim=None,
    ylim=None,
    linestyle='auto',
):
    """
    Plot net stored charge Q vs. -Voltage from SCF simulation results.

    Parameters
    ----------
    CAPS : list of Cap
        List of SCF results.
    label : str
        Legend label.
    ax : matplotlib Axes or None
        If None, a new figure/axes is created.
    color : str or None
        Color for the curve.
    fill : bool
        Whether markers should be filled (only used for marker mode).
    title : str or None
        Title for the axes.
    xlim, ylim : tuple or None
        Axis limits.
    linestyle : str
        - 'auto'    → default style (line + markers)
        - 'line'    → force line only (no markers)
        - 'markers' → force line + circular markers
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import cycle

    # Extract data
    ThetaNa = [c.thetaNa_exc for c in CAPS]
    ThetaCl = [c.thetaCl_exc for c in CAPS]
    Theta = np.array(ThetaNa) - np.array(ThetaCl)

    c0 = CAPS[0]
    unit = c0.electron / (c0.sigma**2)   # C/m^2 per unit theta
    Theta = Theta * unit

    Voltage = np.array([c.V for c in CAPS])

    # Setup axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if color is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = next(cycle(color_cycle))

    # Marker face color
    markerface = color if fill else 'none'

    # Decide plotting style
    if linestyle == 'line':
        # line only
        ax.plot(
            -Voltage, Theta,
            linestyle='-',
            linewidth=2.0,
            color=color,
            label=label,
        )
    else:
        # either 'markers' or 'auto'
        ax.plot(
            -Voltage, Theta,
            marker='o',
            markersize=6,
            markerfacecolor=markerface,
            color=color,
            linestyle='-',
            linewidth=0.8,
            alpha=0.9,
            label=label,
        )

    # Labels
    ax.set_xlabel(r'$\mathrm{Surface\ potential}\ (-\psi),\ \mathrm{V}$')
    ax.set_ylabel(r'$\mathrm{Stored\ charge}\ Q\ (\mathrm{C/m}^2)$')

    # Limits
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if title:
        ax.set_title(title)

    if label:
        ax.legend(frameon=False)

    return fig, ax

def plotTV(CAPS, label='', ax=None, color=None, fill=False, title=None, xlim=None, ylim=None ):
    """
    Plot theta_Na vs. -Voltage from SCF simulation results.

    Parameters:
    - CAPS: list of results with `thetaNa_exc` and `V`
    - label: legend label
    - ax: matplotlib Axes to reuse (optional)
    - color: color of markers (optional)
    - fill: whether to fill markers
    - title: title string (optional)

    Returns:
    - fig: matplotlib Figure
    - ax: matplotlib Axes
    - plt: matplotlib.pyplot
    """
    # ax = axs
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import cycle
    default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_iter = cycle(default_color_cycle)
    
    

    ThetaNa = [c.thetaNa_exc for c in CAPS]
    ThetaCl = [c.thetaCl_exc for c in CAPS]
    Theta = np.array(ThetaNa) - np.array(ThetaCl)
    c = CAPS[0]
    unit = c.electron/(c.sigma**2) # Coulumb /m2
    Theta*=unit
    Voltage = [c.V for c in CAPS]
    Phi = [c.phi for c in CAPS]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    if color is None:
        color = next(color_iter)

    facecolor = color if fill else 'none'

    # ax.scatter(-np.array(Voltage), Theta, facecolor=facecolor, edgecolors=color, label=label)
    # Line + symbols
    markerface = color if fill else 'none'

    ax.plot(-np.array(Voltage), Theta, marker='o', markersize=6,
            markerfacecolor=markerface, 
            # markeredgecolor=color,
            color=color, label=label, linestyle='-',
            linewidth=0.2, alpha=0.8)
    ax.set_xlabel(r'$\textrm{Surface potential } (-\psi), \textrm{V}$')
    # ax.set_ylabel(r'$\theta_{\mathrm{Na}}$')
    ax.set_ylabel(r'$\textrm{Stored charge (Q), C/m}^2$')

    
    
    if xlim:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])

    if title:
        ax.set_title(title)
    if label:
        ax.legend()

    return fig, ax

    
if __name__ == '__main__':

    # values of grafting densities , chains/nm2
    phis = [0.0, 0.2, ]

    # values of pK 
    # pKs = [1.0, 2.0, 3.0]
    pKs = [1.0]

    # salinities in mol/L
    # cnas = [0.1, 0.4, 1.6]
    cnas = [0.4, 1.6]

    # values of Flory-Haggins parameter, in units of kT
    chis = [4.0, 2.0, 0.0]

    CAPS = []
    for pK in pKs :
        paramsHP.update({'pK': pK}) 
        for cna in cnas:
            paramsHP.update({'cna': cna})    
            for chi in chis:
                paramsHP.update({'chi': chi})
                paramsHP.update({'chi_subano': 5.0})
                for phi in PHIs:
                    paramsHP.update({'phi': phi})    

                    for alpha in Alphas:
                        paramsHP.update({'alpha': alpha})    
                        # pprint.pprint(paramsHP)
                        Brush = Cap(**paramsHP)
                        CAPS.append(Brush)

    Brushes_mp(CAPS)
