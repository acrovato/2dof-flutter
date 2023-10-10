from utils.coupling import FlutterGroup
from utils.mapper import PlateMapper
from utils.structures import PlateStructuralSolver
from utils.aerodynamics import PlateAerodynamicSolver
from utils.flutter import NIPK
import openmdao.api as om
import numpy as np

class PitchPlunge(om.Group):
    """Aerostructural flutter model of a pitch-plunge flat plate
    """
    def initialize(self):
         self.options.declare('vrb', default=0, desc='verbosity level')

    def setup(self):
        # Constant parameters
        rho_air = 1.225 # air density
        rho_alu = 2700 # aluminum density
        chord = 0.25 # chord of plate
        freq_n = [2, 8] # natural frequencies of plate
        k_ref = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0]) # reference reduced frequencies
        self.u = np.linspace(5, 55, 51, endpoint=True) # airspeed range
        rho_ks = 1e6 # KS aggregation parameter
        # Solvers
        struct_sol = PlateStructuralSolver(rho_alu, chord, freq_n)
        aero_sol = PlateAerodynamicSolver(chord, k_ref)
        flutter_sol = NIPK(k_ref, 0.5*chord, rho_air, self.u, rho_ks, self.options['vrb'])
        # Components
        self.add_subsystem('dvs', om.IndepVarComp(), promotes=['*'])
        self.add_subsystem('mapper', PlateMapper(), promotes=['*'])
        self.add_subsystem('flutter', FlutterGroup(struct=struct_sol, aero=aero_sol, flutter=flutter_sol))

    def configure(self):
        # DV
        self.dvs.add_output('x_f', val=0.5, desc='x-fraction of flexural axis')
        self.add_design_var('x_f', lower=0.3, upper=0.7, scaler=1)
        self.dvs.add_output('t', val=2e-2, desc='plate thickness')
        self.add_design_var('t', lower=1e-2, upper=2e-2, scaler=1e1)
        for out in self.mapper.list_outputs(out_stream=None):
            self.connect(f'{out[0]}', f'flutter.{out[0]}')
        # CON
        self.add_constraint('flutter.ks_g', upper=-0.001, scaler=1e2)
        # OBJ
        self.add_objective('flutter.m', scaler=1e-2)

def parseargs():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('task', help='slect the task to be performed', default='optimization', const='optimization', nargs='?', choices=['check', 'analysis', 'optimization', 'space'])
    parser.add_argument('-v', help='increase output verbosity', action='count', default=0)
    return parser.parse_args()

def print_time(selapsed):
    days, rem = divmod(selapsed, 24*60*60)
    hours, rem = divmod(rem, 60*60)
    minutes, seconds = divmod(rem, 60)
    print('Wall-clock time: {:0>2}-{:0>2}:{:0>2}:{:0>2}'.format(int(days),int(hours),int(minutes),int(seconds)))

if __name__ == "__main__":
    import time
    # Set up the problem
    args = parseargs()
    prob = om.Problem(model=PitchPlunge(vrb=args.v), driver=om.ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-6, disp=True, maxiter=10))
    prob.model.recording_options['includes'] = ['x_f', 't', 'flutter.f', 'flutter.g', 'flutter.ks_g', 'flutter.m']
    prob.model.add_recorder(om.SqliteRecorder('cases.sql'))
    prob.setup()
    om.n2(prob, show_browser=False, outfile='n2.html')

    # Run
    tic = time.perf_counter()
    if args.task == 'check':
        prob.run_model()
        prob.check_partials(compact_print=False, method='fd', step=1e-6, form='central')
    elif args.task == 'analysis':
        prob.run_model()
    elif args.task == 'optimization':
        prob.run_driver()
    elif args.task == 'space':
        x_range = [0.3, 0.4, 0.5, 0.6, 0.7]
        t_range = [0.01, 0.0125, 0.015, 0.0175, 0.02]
        f_speed = np.zeros((len(x_range), len(t_range)))
        f_mode = np.zeros((len(x_range), len(t_range)))
        for i, x in enumerate(x_range):
            for j, t in enumerate(t_range):
                prob.set_val('x_f', x)
                prob.set_val('t', t)
                prob.run_model()
                f_speed[i,j] = prob.get_val('flutter.flutter.f_speed')
                f_mode[i,j] = prob.get_val('flutter.flutter.f_mode')
    else:
        raise RuntimeError(f'task {args.task} not defined!\n')
    toc = time.perf_counter()
    print_time(toc-tic)

    # Print the history
    if args.task != 'space':
        cases = om.CaseReader('cases.sql').get_cases()
        print('{:>10s}, {:>10s}, {:>10s}, {:>10s}, {:>10s}'.format('it', 'x_f', 't', 'ks_g', 'm'))
        for i, case in enumerate(cases):
            xf = case.get_design_vars(scaled=False)['x_f'][0]
            t = case.get_design_vars(scaled=False)['t'][0]
            ks = case.get_constraints(scaled=False)['flutter.ks_g'][0]
            m = case.get_objectives(scaled=False)['flutter.m'][0]
            print('{:10d}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}'.format(i, xf, t, ks, m))

    # Parameters space plots
    if args.task == 'space':
        import matplotlib.pyplot as plt
        t_range, x_range = np.meshgrid(t_range, x_range)
        mode = ['red'] * len(x_range) * len(t_range)
        for i in range(len(x_range)):
            for j in range(len(t_range)):
                if f_mode[i,j] < 2:
                    mode[i*len(t_range) + j] = f'C{int(f_mode[i,j])}'
        ax = plt.axes(projection='3d')
        ax.plot_wireframe(x_range, t_range, f_speed, color='gray', rstride=1, cstride=1)
        ax.scatter(x_range, t_range, f_speed, c=mode, marker='o')
        ax.set_xlabel('Torsion center, $x_f$')
        ax.set_ylabel('Thickness, $t$')
        ax.set_zlabel('Flutter speed, $u$')
        plt.savefig('space.pdf') # plt.show()
    # V-f-g plots
    if args.task == 'analysis' or args.task == 'optimization':
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 1)
        u = prob.model.u
        modes = ['plunge', 'pitch']
        for j in range(2):
            # Frequency
            axs[0].plot(u, cases[0].get_val('flutter.f')[:,j], label=f'{modes[j]} - init', color=f'C{j}', lw=2, ls='--')
            axs[0].plot(u, cases[-1].get_val('flutter.f')[:,j], label=f'{modes[j]} - opti', color=f'C{j}', lw=2, ls='-')
            axs[0].set_xlabel('Airspeed, $u$')
            axs[0].set_ylabel('Frequency, $f$ (Hz)')
            axs[0].legend()
            # Damping
            axs[1].plot(u, cases[0].get_val('flutter.g')[:,j], label=f'{modes[j]} - init', color=f'C{j}', lw=2, ls='--')
            axs[1].plot(u, cases[-1].get_val('flutter.g')[:,j], label=f'{modes[j]} - opti', color=f'C{j}', lw=2, ls='-')
            axs[1].set_xlabel('Airspeed, $u$')
            axs[1].set_ylabel('Damping, $g$')
            axs[1].legend()
        axs[1].plot(u, [0.] * len(u), color='k', lw=1, ls='-.') # zero-damping line
        fig.tight_layout()
        plt.savefig('v_f_g.pdf') # plt.show()
