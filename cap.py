import os
import glob
import time
import subprocess
import pickle  # keep if you still want to pickle Cap objects; otherwise can be removed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shortuuid

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})


class Cap:
    """
    Wrapper around SFBox input generation, execution, and post-processing
    for a single brush-modified half-cell configuration.
    """

    # Physical constants and SFBox units (class-level)
    val = -1.0              # valence of charged polymer state (A1)
    epsilon = 80            # dielectric permittivity of water

    lB = 0.7e-9             # [m] Bjerrum length in water at 300K
    sigma = lB / 2.0        # [m] lattice spacing in SFBox model

    Navogadro = 6.022e23    # [1/mol] Avogadro number
    electron = 1.60217663e-19  # [C] elementary charge
    kT = 4.14e-21           # [J] 1 kT at 300K

    iguess_in = ""
    iterationlimit = 20000
    PATH = "data"           # directory where SFBox input/output are stored
    water_autoprotolysis = True

    def __init__(
        self,
        D=500,               # number of lattice layers
        H=10,                # thickness of dielectric / electrode-support layer
        cna=1e-1,            # bulk NaCl concentration (mol/L)
        phi=0.01,            # grafting density (chains / nm^2)
        N=1,                 # chain length
        chi=0.0,             # polymer–water chi
        chi_subano=5.0,      # substrate–water chi
        chi_salt=0.0,        # salt–polymer chi (optional)
        epsilon=80,        # dielectric constant in solution (default 80)
        epsilon_subano=80, # dielectric constant in substrate (default 80)
        poly_epsilon=80,   # optional dielectric constant of polymer (default 80)
        pK=1.0,              # redox pK (larger => more charge)
        pKdop=None,           # Na+ “doping” pK (0 => disabled)
        alpha=0.5,           # alphabulk for Ai monomers
        pKw=14.0,            # water autoprotolysis constant (pKw)
        iguess_in="",
        alpha_s=None,        # substrate valence (default = -alpha)
        sigma=None,          # optional override of lattice spacing
        suffix="",
        timeout=200,         # SFBox timeout [s]
    ):
        # Geometry and composition
        self.D = D
        self.H = H
        self.N = N
        self.chi = chi
        self.chi_subano = chi_subano
        self.chi_salt = chi_salt
        self.epsilon = epsilon
        self.epsilon_subano = epsilon_subano
        self.phi = phi
        self.cna = cna
        self.pK = pK
        self.pKdop = pKdop

        # Redox / charge parameters
        self.alpha = alpha
        if not self.alpha:
            # print("alpha = 0 → no redox possible, setting pK = 0.")
            self.pK = 0.0

        self.alpha_s = alpha_s if alpha_s is not None else -alpha

        # Override lattice spacing if requested
        if sigma is not None:
            self.sigma = sigma

        # Unit conversions
        self.to_mols = (self.Navogadro * self.sigma**3 * 1000.0) ** (-1)  # [mol/L]
        self.to_chainspernm2 = (self.sigma * 1e9) ** (-2)                 # [chains/nm^2]
        self.to_bars = self.kT / self.sigma**3 / 1e5                      # [bar]
        self.to_coulombsperm2 = self.electron / (self.sigma**2)           # [C/m^2]

        # Maximum solubility (NaCl; can be adjusted)
        cs_max = 6.1       # mol/L
        self.cs_max = cs_max / self.to_mols

        # Reference water autoprotolysis (for SFBox units)
        self.pKw_default = -np.log10(1e-14 / self.to_mols**2)

        # Convert grafting density and salt to SFBox volume fractions
        self.phi_sf = phi / self.to_chainspernm2
        self.cna_sf = cna / self.to_mols

        # Water autoprotolysis pKw in SFBox units
        self.pKw = -np.log10(10 ** (-pKw) / self.to_mols**2)
        self.pKw = np.round(self.pKw, 2)

        # Initial guess and run control
        self.iguess_in = iguess_in
        self.suffix = suffix
        self.timeout = timeout
        self.poly_epsilon = poly_epsilon

        # Internal state
        self.input = []
        self.solved = False
        self.datapro = None
        self.datakal = None
        self.profiles = None
        self.V = 0.0  # potential drop

        self.__str__()  # initialize filenames

    # ------------------------------------------------------------------
    # Naming and filenames
    # ------------------------------------------------------------------
    def __str__(self):
        """Construct a unique descriptive name for this configuration."""
        self.name = "Cap"
        self.name += f"_N{self.N}"
        self.name += f"_chi{self.chi}"
        self.name += f"_chi_subano{self.chi_subano}" * bool(self.chi_subano)
        self.name += f"_chi_salt{self.chi_salt}" * bool(self.chi_salt) * bool(self.phi)
        self.name += f"_epsilon{self.epsilon}"* bool(self.epsilon-80)
        self.name += f"_epsilon_subano{self.epsilon_subano}" * bool(self.epsilon_subano-80)
        self.name += f"_poly_epsilon{self.poly_epsilon}" * bool(self.poly_epsilon-80)

        self.name += f"_phi{self.phi:.3f}"
        self.name += f"_cna{self.cna:.4f}"
        self.name += f"_pK{self.pK}" * bool(self.phi)
        self.name += f"_pKdop{self.pKdop}" * bool(self.pKdop) * bool(self.phi)

        self.name += f"_alpha{self.alpha:.3f}"
        self.name += f"_alphas{self.alpha_s:.3f}" * bool(self.alpha_s-self.alpha)
        self.name += f"_D{self.D}_H{self.H}"
        self.name += f"_sigma{self.sigma:.3e}"

        self.name += "_ig" * bool(self.iguess_in)

        # Unique file prefix (shortuuid seeded with name)
        self.fname = "Cap" + shortuuid.uuid(self.name)

        self.fnamein = self.fname + ".in"
        self.fnamepro = self.fname + ".pro"
        self.fnamekal = self.fname + ".kal"
        self.fnameout = self.fname + ".ana"
        self.fnameiguess = self.fname + ".ig"
        self.fnamepkl = self.fname + ".pkl"

        return self.name

    # ------------------------------------------------------------------
    # Building SFBox input
    # ------------------------------------------------------------------
    def addLattice(self):
        self.input.append("\n//////////////// addLattice ///////////////")
        self.input.append("lat : mylat : geometry : flat")
        self.input.append("lat : mylat : lambda : 0.166666666666666666666")
        self.input.append("lat : mylat : lowerbound : surface")
        self.input.append(f"lat : mylat : bondlength : {self.sigma}")
        self.input.append(f"lat : mylat : n_layers : {self.D}")

    def addOutput(self):
        self.input.append("\n//////////////// addOutput ///////////////")
        self.input.append(f"output : {self.fnameout} : type : ana")
        self.input.append(f"output : {self.fnameout} : append : false")
        self.input.append(f"output : {self.fnamepro} : type : profiles")
        self.input.append(f"output : {self.fnamekal} : type : kal")
        self.input.append("sys : noname : overflow_protection : true")
        self.input.append(f"newton : isac : iterationlimit : {self.iterationlimit}")
        self.input.append("newton : isac : tolerance : 1e-7")
        self.input.append(f"newton : isac : initial_guess_output_file : {self.fnameiguess}")
        if self.iguess_in and os.path.exists(os.path.join(self.PATH, self.iguess_in)):
            self.input.append("newton : isac : initial_guess : file")
            self.input.append(f"newton : isac : initial_guess_input_file : {self.iguess_in}")

    def addAnode(self):
        """Define dielectric slab and charged substrate (subano) at z in [H..H]."""
        self.input.append("//////////////// addDielectricLayer ///////////////")
        self.input.append("mon : dielectric : freedom : frozen")
        self.input.append(f"mon : dielectric : frozen_range : 0;{self.H - 1}")

        self.input.append("//////////////// addSubstrate ///////////////")
        self.input.append("mon : subano : freedom : frozen")
        self.input.append(
            f"mon : subano : frozen_range : {self.H};{self.H}"
        )
        self.input.append(f"mon : subano : valence :    {self.alpha_s}")
        self.input.append(f"mon : subano : chi - water : {self.chi_subano}")
        self.input.append(f"mon : subano : epsilon : {self.epsilon_subano}")
        self.input.append("")

    def addBrush(self):
        """Define the grafted brush and redox reactions along the chain."""
        self.input.append("//////////////// addBrush ///////////////")

        if self.N == 1:
            composition = "(Ai)1"
        elif self.N == 2:
            composition = "(Ai)1(Ae)1"
        else:
            composition = f"(Ai)1(A){self.N - 2}(Ae)1"

        import re

        monomers = re.findall(r"\(.*?\)", composition)
        monomers = [m[1:-1] for m in monomers]

        # Grafting segment pinned at z = H+1
        self.input.append("mon : Ai : freedom : pinned")
        self.input.append(f"mon : Ai : pinned_range : {self.H + 1};{self.H + 1}")

        # Polymer monomers
        for mon in monomers:
            if mon[0] == "A":
                chi_val = self.chi
                pK_val = self.pK
            else:
                raise ValueError(f"Unknown monomer: {mon}")

            self.input.append(f"mon : {mon} : chi - water : {chi_val}")

            if self.poly_epsilon:
                self.input.append(f"mon : {mon} : epsilon : {self.poly_epsilon}")

            if self.pK:
                self.input.append(f"mon : {mon} : state1 : {mon}1")
                self.input.append(f"mon : {mon} : state2 : {mon}0")

                self.input.append(f"state : {mon}0 : valence : 0")
                self.input.append(f"state : {mon}1 : valence : {self.val}")

                if mon == "Ai":
                    self.input.append(f"state : Ai1 : alphabulk : {self.alpha}")
                else:
                    reactionstring = f"reaction : {mon} : equation : 1({mon}1)1(Ai0)=1({mon}0)1(Ai1)"
                    self.input.append(reactionstring)
                    self.input.append(f"reaction : {mon} : pK : {pK_val}")

        # Brush molecule
        self.input.append("mol : Brush : freedom : restricted")
        self.input.append(f"//// phi = {self.phi}")
        self.input.append(f"mol : Brush : theta : {self.phi_sf * self.N}")
        self.input.append("mol : Brush : composition : " + composition)
        self.input.append("")

    def addWater(self):
        """Define water, Na+, Cl-, and water autoprotolysis."""
        self.input.append("//////////////// addWater ///////////////")
        self.input.append("mon : water : freedom : free")
        self.input.append(f"mon : water : epsilon : {self.epsilon}")

        self.input.append("mon : cl : freedom : free")
        self.input.append("mon : na : freedom : free")

        self.input.append("mol : Water : composition : (water)1")
        self.input.append("mol : Water : freedom : solvent")

        self.input.append("mol : Cl : composition  : (cl)1")
        self.input.append("mol : Na : composition : (na)1")

        self.input.append("mol : Cl : freedom : free")
        self.input.append(f"mol : Cl : phibulk : {self.cna_sf:.5f}")
        self.input.append("mol : Na : freedom : free")
        self.input.append(f"mol : Na : phibulk : {self.cna_sf:.5f}")

        if self.water_autoprotolysis:
            self.input.append("mon : water : state1 : H3O")
            self.input.append("mon : water : state2 : H2O")
            self.input.append("mon : water : state3 : OH")
            self.input.append("state : H3O : valence : 1")
            self.input.append("state : H2O : valence : 0")
            self.input.append("state : OH : valence : -1")
            self.input.append("reaction : auto_w : equation : 2(H2O)=1(OH)1(H3O)")
            self.input.append(f"reaction : auto_w : pK : {self.pKw}")

            self.input.append("mon : cl : valence : -1")
            self.input.append("mon : na : valence : 1")

    def inputfilegen(self, verbose = False):
        """Generate and write SFBox input file."""
        str(self)  # ensure filenames are up to date
        self.input = []
        self.addAnode()
        if self.phi:
            self.addBrush()
        self.addWater()
        self.addLattice()
        self.addOutput()

        os.makedirs(self.PATH, exist_ok=True)
        infile_path = os.path.join(self.PATH, self.fnamein)
        with open(infile_path, "w") as infile:
            for line in self.input:
                infile.write(line + "\n")
        if verbose:
            print("Input file saved to:", infile_path)

    # ------------------------------------------------------------------
    # Running SFBox and loading data
    # ------------------------------------------------------------------
    def check4timeout(self):
        """
        Check if this input has previously timed out.
        Looks for files: <fnamein>.timeout<timeout>.
        """
        pattern = os.path.join(self.PATH, self.fnamein + ".timeout*")
        matches = glob.glob(pattern)
        if not matches:
            return False

        timeouts = []
        for match in matches:
            suffix = match[len(f"{self.PATH}/{self.fnamein}.timeout") :]
            try:
                timeouts.append(int(suffix))
            except ValueError:
                continue

        return bool(timeouts and np.max(timeouts) >= self.timeout)

    def sfbox_run(self, verbose=False):
        """
        Run the 'sfbox' binary with the generated input file using subprocess.
        If the process times out, kill it and tag the input file with ".timeout<timeout>".
        """
        CWD = os.getcwd()
        os.chdir(self.PATH)
        start_time = time.time()
        print(f"{self.fname}: running\n")

        proc = None
        try:
            proc = subprocess.Popen(
                ["sfbox", self.fnamein],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if verbose:
                print(f"[{self.fname}] Started with PID {proc.pid}")

            try:
                self.stdout, self.stderr = proc.communicate(timeout=self.timeout)
                elapsed = time.time() - start_time
                if verbose:
                    print(f"[{self.fname}] Finished in {elapsed:.2f} s, return code {proc.returncode}")
                self.solved = (proc.returncode == 0)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.stdout, self.stderr = proc.communicate()
                self.solved = False
                if verbose:
                    print(f"[{self.fname}] TIMEOUT {self.timeout} s → killed.")
                os.rename(self.fnamein, self.fnamein + f".timeout{self.timeout}")

        except Exception as e:
            self.stdout = ""
            self.stderr = str(e)
            self.solved = False
            print(f"[{self.fname}] Launch error: {e}")

        os.chdir(CWD)
        # print()
        return proc.returncode if proc else 1

    def loadData(self, silent=True):
        """
        Ensure that SFBox has been run and load .pro and .kal files.
        If data exist, they are reused; otherwise SFBox is launched.
        """
        self.solved = False

        if self.check4timeout():
            if not silent:
                print(f"{self.fnamein}: timed out in a previous attempt.")
            return

        pro_path = os.path.join(self.PATH, self.fnamepro)
        kal_path = os.path.join(self.PATH, self.fnamekal)

        if os.path.exists(pro_path) and os.path.exists(kal_path):
            if not silent:
                print(f"{self.fname}: data already available.")
            self.solved = True
            self.datapro = pd.read_csv(pro_path, delimiter="\t")
            self.datakal = pd.read_csv(kal_path, delimiter="\t")
        else:
            if not silent:
                print(f"No data found for {self.fname}. Running SFBox ...")
            self.inputfilegen()
            return_code = self.sfbox_run()
            if return_code or not os.path.exists(pro_path):
                self.solved = False
                if not silent:
                    print(f"Simulation failed for {self.fnamein}")
            else:
                self.solved = True
                self.datapro = pd.read_csv(pro_path, delimiter="\t")
                self.datakal = pd.read_csv(kal_path, delimiter="\t")

        if self.solved:
            # Clean column names to be Python-friendly
            translate_table = str.maketrans(":-", "__", " ")
            self.datakal.columns = self.datakal.columns.str.translate(translate_table)
            self.datapro.columns = self.datapro.columns.str.translate(translate_table)

        if not silent:
            print(f"{self.fnamein} → {'Solved' if self.solved else 'Failed'}")

    # ------------------------------------------------------------------
    # Post-processing: profiles & scalars
    # ------------------------------------------------------------------
    def getProfiles(self):
        """
        Build a dict of 1D profiles (mol/L where applicable) from .pro.
        Also computes electrode-to-bulk potential drop self.V.
        """
        profiles = {}
        to_mols = self.to_mols

        if self.alpha_s:
            profiles["subano"] = np.abs(
                self.datapro.mon_subano_phi.to_numpy() * self.datakal.mon_subano_valence.to_numpy() * to_mols
            )
        else:
            profiles["subano"] = np.zeros(self.D + 2)

        profiles["Cl"] = self.datapro.mol_Cl_phi * to_mols
        profiles["Na"] = self.datapro.mol_Na_phi * to_mols

        profiles["H3O"] = self.datapro.state_H3O_phi * to_mols
        profiles["OH"] = self.datapro.state_OH_phi * to_mols
        profiles["H2O"] = self.datapro.state_H2O_phi * to_mols

        profiles["potential"] = self.datapro.sys_noname_potential

        # Polymer-related profiles
        profiles["Brush"] = np.zeros(self.D + 2)
        profiles["Ae"] = np.zeros(self.D + 2)
        profiles["A1"] = np.zeros(self.D + 2)
        profiles["A0"] = np.zeros(self.D + 2)
        profiles["ANa"] = np.zeros(self.D + 2)
        profiles["charged"] = np.zeros(self.D + 2)

        if self.phi:
            profiles["Brush"] += self.datapro.mol_Brush_phi * to_mols

            # chain ends
            try:
                profiles["Ae"] += self.datapro.mon_Ae_phi * to_mols
            except AttributeError:
                pass

            # charged segments
            for col in ["state_Ai1_phi", "state_A1_phi", "state_Ae1_phi"]:
                if col in self.datapro.columns:
                    profiles["A1"] += self.datapro[col] * to_mols

            # neutral segments
            for col in ["state_Ai0_phi", "state_A0_phi", "state_Ae0_phi"]:
                if col in self.datapro.columns:
                    profiles["A0"] += self.datapro[col] * to_mols

            profiles["charged"] = profiles["A1"]

            # Na+-condensed segments (if doping enabled)
            if self.pKdop:
                for col in ["state_AiNa_phi", "state_ANa_phi", "state_AeNa_phi"]:
                    if col in self.datapro.columns:
                        profiles["ANa"] += self.datapro[col] * to_mols

            profiles["epsilon"] = self.datapro.sys_noname_epsilon

        profiles["sites"] = np.ones(self.D + 2)

        self.profiles = profiles
        self.V = profiles["potential"].iloc[0] - profiles["potential"].iloc[-1]
        return profiles

    def getTheta(self):
        """
        Extract integrated theta values (total amounts) for main species.
        Used mainly for annotation in getPlotTitle().
        """
        self.thetaBrush = 0
        self.thetaA = 0
        self.thetaA0 = 0
        self.thetaA1 = 0

        if self.phi:
            self.thetaBrush += float(self.datakal.mol_Brush_theta.iloc[0])
            self.thetaA += float(self.datakal.mon_Ai_theta.iloc[0])
            self.thetaA += float(self.datakal.mon_Ae_theta.iloc[0])
            if self.N > 2:
                self.thetaA += float(self.datakal.mon_A_theta.iloc[0])

            if self.pK:
                if self.N > 2:
                    self.thetaA0 += float(self.datakal.state_A0_theta.iloc[0])
                    self.thetaA1 += float(self.datakal.state_A1_theta.iloc[0])
                self.thetaA0 += float(self.datakal.state_Ae0_theta.iloc[0])
                self.thetaA0 += float(self.datakal.state_Ai0_theta.iloc[0])
                self.thetaA1 += float(self.datakal.state_Ae1_theta.iloc[0])
                self.thetaA1 += float(self.datakal.state_Ai1_theta.iloc[0])

        # water ions
        try:
            self.thetaH3O = float(self.datakal.state_H3O_theta.iloc[0])
            self.thetaH3O_exc = float(self.datakal.state_H3O_thetaexcess.iloc[0])
        except KeyError:
            self.thetaH3O = 0
            self.thetaH3O_exc = 0

        try:
            self.thetaOH = float(self.datakal.state_OH_theta.iloc[0])
            self.thetaOH_exc = float(self.datakal.state_OH_thetaexcess.iloc[0])
        except KeyError:
            self.thetaOH = 0
            self.thetaOH_exc = 0

        # salt
        try:
            self.thetaNa = float(self.datakal.mol_Na_theta.iloc[0])
            self.thetaNa_exc = float(self.datakal.mol_Na_thetaexcess.iloc[0])
        except KeyError:
            self.thetaNa = 0
            self.thetaNa_exc = 0

        try:
            self.thetaCl = float(self.datakal.mol_Cl_theta.iloc[0])
            self.thetaCl_exc = float(self.datakal.mol_Cl_thetaexcess.iloc[0])
        except (KeyError, AttributeError):
            self.thetaCl = 0
            self.thetaCl_exc = 0

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def getPlotTitle(self):
        """Build a LaTeX-formatted multi-line title summarizing the state."""
        self.getTheta()

        title  = r"\noindent"
        title += rf"$\alpha = {self.alpha * self.to_coulombsperm2:.2f}\,\mathrm{{C/m}}^2"
        title += rf"\;({self.alpha:.2f})$\\"

        title += rf"$U = {self.V:.3f}\,\mathrm{{V}}$\\"

        title += rf"$\phi = {self.phi}\,\mathrm{{chains/nm}}^2"
        title += rf"\;({self.phi_sf:.4f})$\\"

        title += rf"$c_s = {self.cna}\,\mathrm{{mol/L}}"
        title += rf"\;({self.cna_sf:.4f})$\\"

        title += rf"$\chi = {self.chi}\,kT$\\"

        title += (
            rf"$\theta_{{\mathrm{{Na}}}}^{{\mathrm{{ex}}}}"
            rf"= {self.thetaNa_exc * self.to_coulombsperm2:.3f}\,\mathrm{{C/m}}^2,"
        )
        title += (
            rf"\;\theta_{{\mathrm{{Cl}}}}^{{\mathrm{{ex}}}}"
            rf"= {self.thetaCl_exc * self.to_coulombsperm2:.3f}\,\mathrm{{C/m}}^2$"
        )

        return title

    def plotSelectedProfiles(
        self,
        selected_lines=['Brush', 'charged'],
        # selected_lines=['Brush', 'charged', 'Ae', 'A1', 'Be', 'B1', 'Na', 'Cl',  'potential', 'subano'],
        y_scale='log',
        y_min=1e-3,
        x_max=40,
        x_min=None,
        ax=None,                         # Accept external axis
        add_title=True,                  # Optional: control title placement
        show=False,                       # Optional: avoid plt.show() in grid
    ):
        if not selected_lines:
            print("No lines selected for plotting.")
            return

        # self.loadData()
        if not self.solved:
            return

        # profiles = self.getProfiles()
        profiles = self.profiles
        X = (np.arange(self.D + 2) - self.H )* self.sigma / 1e-9

        lines = {
            'Brush':    [profiles['Brush'],     {'color': 'black', 'linestyle': 'solid', 'marker': 'o', 'markersize':5.0, 'alpha':0.6, 'linewidth':6,'label':"Brush"}],
            'charged':  [profiles['charged'],   {'color': 'black', 'linestyle': 'solid', 'alpha':1.0, 'linewidth':2,'label':"charged"}],
            'Ae':       [profiles['Ae'],        {'color': 'green', 'linestyle': 'solid', 'alpha':0.5, 'linewidth':4,'label':"Ae"}],
            'A1':       [profiles['A1'],        {'color': 'green', 'alpha':1.0, 'linewidth':4,'label':"A1"}],
            'A0':       [profiles['A0'],        {'color': 'green', 'alpha':0.5, 'linewidth':4,'label':"A0"}],
            'ANa':      [profiles['ANa'],       {'color': 'pink', 'alpha':0.5, 'linewidth':4,'label':"ANa"}],
            'Na':       [profiles['Na'],        {'color': 'red', 'alpha': 1.0, 'linestyle': 'solid', 'linewidth':2, 'label':"Na"}],
            'Cl':       [profiles['Cl'],        {'color': 'blue','alpha': 1.0, 'linestyle': 'solid', 'linewidth':2, 'label':"Cl"}],
            'H3O':      [profiles['H3O'],       {'color': 'red', 'linestyle': 'solid','label':"H3O"}],
            'H2O':      [profiles['H2O'],       {'color': 'grey', 'linestyle': 'solid','label':"H2O"}],
            'OH':       [profiles['OH'],        {'color': 'blue', 'linestyle': 'solid','label':"OH"}],
            'subano':   [profiles['subano'],    {'color': 'black', 'linestyle': 'solid','linewidth':8,'label':"subano"}],
            'potential':[profiles['potential'], {'color': 'grey', 'label':"$\\Psi$"}],
        }

        if self.N > 1:
            lines['Ae'][1]['linestyle'] = 'dashed'
            

        # Use provided axis or create new one
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4.0), constrained_layout=True)
        else:
            fig = ax.get_figure()          

        # --- main plots ---
        for line in selected_lines:
            if line != 'potential' and line in lines:
                ax.plot(X, lines[line][0], **lines[line][1])

        ax.set_yscale(y_scale)
        ax.set_ylim(bottom=y_min, top=1.1 * self.to_mols)
        ax.set_xlim(right=x_max, left = (-self.H * self.sigma / 1e-9))
        if x_min is not None:
            ax.set_xlim(left = x_min)
        ax.set_xlabel(r'$\textrm{Distance } (z), \textrm{nm}$')
        ax.set_ylabel(r'$\textrm{Densities } (\varphi_{i}), \textrm{mol/L}$')
        # ax.grid(True)

        if add_title:
            title = self.getPlotTitle()
            ax.text(
                0.8, 0.45, title,
                transform=ax.transAxes,
                fontsize=12,
                color="black",
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4', alpha=0.8)
            )

        if 'potential' in selected_lines:
            ax2 = ax.twinx()
            ax2.plot(X, lines['potential'][0], 'k--', label='Potential')
            ax2.set_ylabel(r'$\textrm{Potential } (\psi), \textrm{V}$', color='black')
            ax2.set_ylim(bottom=-0.6, top=0.1)
            ax2.tick_params(axis='y', labelcolor='black')

        if show:
            plt.tight_layout()
            plt.show()
            output_pdf_path = os.path.join(f"figures/{self.fname}.pdf")
            output_svg_path = os.path.join(f"figures/{self.fname}.svg")
            plt.savefig(output_pdf_path, format='pdf')
            plt.savefig(output_svg_path, format='svg')
            print(f'Figure saved to {output_pdf_path}')
        else: output_pdf_path =''
        return ax, output_pdf_path # Always return the axis


    # ------------------------------------------------------------------
    # (Optional) simple pickle helpers
    # ------------------------------------------------------------------
    def to_pickle(self):
        pklfile = os.path.join(self.PATH, self.fnamepkl)
        with open(pklfile, "wb") as file:
            pickle.dump(self, file)
        print(f"Object pickled to {pklfile}")

    @classmethod
    def from_pickle(cls, pklfile):
        with open(pklfile, "rb") as file:
            obj = pickle.load(file)
        return obj


if __name__ == "__main__":
    import pprint

    # Example: single brush configuration and profile plot
    params = {
        "N": 200,
        "phi": 0.4,
        "cna": 0.4,
        "alpha": 0.5,
        "pK": 1.0,
        "timeout": 600,
        "chi": 0.0,
        "chi_subano": 0.0,
        "chi_salt": 0.0,
    }

    pprint.pprint(params)
    brush = Cap(**params)
    brush.loadData(silent=False)           # run SFBox if needed
    if brush.solved:
        brush.getProfiles()
        brush.plotSelectedProfiles(show=True)
    else:
        print("Simulation not solved; no plot generated.")
