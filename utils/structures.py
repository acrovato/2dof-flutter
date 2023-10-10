# -*- coding: utf-8 -*-

# 
# Copyright (C) 2023 Adrien Crovato
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import openmdao.api as om
import numpy as np

class Structures(om.ExplicitComponent):
    """OpenMDAO structural wrapper
    """
    def initialize(self):
        self.options.declare('sol', desc='structural solver', recordable=False)

    def setup(self):
        self.sol = self.options['sol']
        self.add_input('dv_geo', shape_by_conn=True, desc='geometric design variables')
        self.add_input('dv_struct', shape_by_conn=True, desc='structural design variables')
        self.add_output('m', val=1., desc='mass')
        self.add_output('M', val=np.ones((2, 2)), desc='mass matrix')
        self.add_output('K', val=np.ones((2, 2)), desc='stiffness matrix')

    def setup_partials(self):
        self.declare_partials(of=['m', 'M', 'K'], wrt=['dv_struct', 'dv_geo'], method='exact')

    def compute(self, inputs, outputs):
        self.sol.set_dvs(inputs['dv_struct'][0], inputs['dv_geo'][0])
        self.sol.compute()
        outputs['m'] = self.sol.m[0,0]
        outputs['M'] = self.sol.m
        outputs['K'] = self.sol.k

    def compute_partials(self, inputs, partials):
        self.sol.compute_gradients()
        partials['m', 'dv_struct'][0] = self.sol.m_t[0,0]
        partials['m', 'dv_geo'][0] = self.sol.m_x[0,0]
        partials['M', 'dv_struct'][:,0] = self.sol.m_t.flatten()
        partials['M', 'dv_geo'][:,0] = self.sol.m_x.flatten()
        partials['K', 'dv_struct'][:,0] = self.sol.k_t.flatten()
        partials['K', 'dv_geo'][:,0] = self.sol.k_x.flatten()

class PlateStructuralSolver:
    """Structural model of pitch-plunge flat plate
    """
    def __init__(self, rho, chord, freq):
        self._rho = rho # material density
        self._chrd = chord # plate chord
        self._freq = freq # natural frequencies

    def set_dvs(self, thck, x_tor):
        """Set design variables
        """
        self.t = thck # plate thickness
        self.xc = x_tor # center of torsion

    def compute(self):
        """Compute mass and stiffness matrices
        """
        self._m = self._rho * self._chrd * self.t # mass
        s = self._m * self._chrd * (0.5 - self.xc) # static imbalance
        i = self._m * self._chrd**2 * (1/3 - self.xc + self.xc**2) # inertia in pitch
        k_h = self._m * (2 * np.pi * self._freq[0])**2 # structural stiffness in plunge
        k_a = i * (2 * np.pi * self._freq[1])**2 # structural stiffness in pitch
        # Matrices
        self.m = np.array([[self._m, s], [s, i]])
        self.k = np.array([[k_h, 0.], [0., k_a]])

    def compute_gradients(self):
        """Compute derivative of mass and stiffness matrices wrt. design variables
        """
        dm_dt = self._rho * self._chrd
        ds_dt = dm_dt * self._chrd * (0.5 - self.xc)
        di_dt = dm_dt * self._chrd**2 * (1/3 - self.xc + self.xc**2)
        ds_dx = -self._m * self._chrd
        di_dx = self._m * self._chrd**2 * (2 * self.xc - 1)
        dkh_dt = dm_dt * (2 * np.pi * self._freq[0])**2
        dka_dt = di_dt * (2 * np.pi * self._freq[1])**2
        dka_dx = di_dx * (2 * np.pi * self._freq[1])**2
        # Gradients
        self.m_t = np.array([[dm_dt, ds_dt], [ds_dt, di_dt]])
        self.m_x = np.array([[0, ds_dx], [ds_dx, di_dx]])
        self.k_t = np.array([[dkh_dt, 0.], [0., dka_dt]])
        self.k_x = np.array([[0., 0.], [0., dka_dx]])
