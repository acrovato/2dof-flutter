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

class Aerodynamics(om.ExplicitComponent):
    """OpenMDAO aerodynamic wrapper
    """
    def initialize(self):
        self.options.declare('sol', desc='aerodynamic solver', recordable=False)

    def setup(self):
        self.sol = self.options['sol']
        self.add_input('dv_geo', shape_by_conn=True, desc='geometric design variables')
        self.add_output('Q_re', val=np.ones((self.sol.n_k, 2, 2)), desc='real part of generalized aerodynamic forces matrices')
        self.add_output('Q_im', val=np.ones((self.sol.n_k, 2, 2)), desc='imag part of generalized aerodynamic forces matrices')

    def setup_partials(self):
        self.declare_partials(of=['Q_re', 'Q_im'], wrt='dv_geo', method='exact')

    def compute(self, inputs, outputs):
        self.sol.set_dvs(inputs['dv_geo'][0])
        self.sol.compute()
        outputs['Q_re'] = np.real(self.sol.q)
        outputs['Q_im'] = np.imag(self.sol.q)

    def compute_partials(self, inputs, partials):
        self.sol.compute_gradients()
        partials['Q_re', 'dv_geo'][:,0] = np.real(self.sol.q_x).flatten()
        partials['Q_im', 'dv_geo'][:,0] = np.imag(self.sol.q_x).flatten()

class PlateAerodynamicSolver:
    """Aerodynamic model of pitch-plunge flat plate
    """
    def __init__(self, chord, k_ref):
        self._chrd = chord # plate chord
        self._k = k_ref # reference reduced frequencies
        self.n_k = len(k_ref) # number of reference reduced frequencies

    def set_dvs(self, x_tor):
        """Set design variables
        """
        self.xc = x_tor # center of torsion

    def compute(self):
        """Compute generalized aerodynamic matrices at reference reduced frequencies
        """
        # Geometric constants
        self._e = np.array([self.xc - 0.25, 0.75 - self.xc, self.xc - 0.50]) * self._chrd
        # Constant matrix components
        c0 = [- 4 * np.pi * 1j,
              + 2 * np.pi]
        c1 = [- 2 * np.pi * self._chrd,
              - 2 * np.pi * 0.5 * self._chrd * 1j,
              - 4 * np.pi * self._e[1] * 1j,
              - 2 * np.pi * (0.5 * self._chrd)**2]
        c2 = [+ 4 * np.pi * self._e[0] * 1j,
              - 2 * np.pi * self._e[2]]
        c3 = [+ 2 * np.pi * self._e[0] * self._chrd,
              - 2 * np.pi * self._e[1] * 0.5 * self._chrd * 1j,
              + 4 * np.pi * self._e[0] * self._e[1] * 1j,
              + 2 * np.pi * (self._e[2]**2 + 0.125 * (0.5 * self._chrd)**2)]
        # GAF matrices
        self.q = np.zeros((len(self._k), 2, 2), dtype=complex)
        for i, k in enumerate(self._k):
            self.q[i,:,:] = self.__compute_gaf(c0, c1, c2, c3, k)

    def compute_gradients(self):
        """Compute derivative of generalized aerodynamic matrices at reference reduced frequencies wrt. design variables
        """
        # Geometric constants
        de = np.array([1., -1., 1.]) * self._chrd
        # Constant matrix components
        dc0 = [0,
               0]
        dc1 = [0,
               0,
               - 4 * np.pi * de[1] * 1j,
               0]
        dc2 = [+ 4 * np.pi * de[0] * 1j,
               - 2 * np.pi * de[2]]
        dc3 = [+ 2 * np.pi * de[0] * self._chrd,
               - 2 * np.pi * de[1] * 0.5 * self._chrd * 1j,
               + 4 * np.pi * (de[0] * self._e[1] + self._e[0] * de[1]) * 1j,
               + 2 * np.pi * 2 * self._e[2] * de[2]]
        # GAF matrices
        self.q_x = np.zeros((len(self._k), 2, 2), dtype=complex)
        for i, k in enumerate(self._k):
            self.q_x[i,:,:] = self.__compute_gaf(dc0, dc1, dc2, dc3, k)

    def __compute_gaf(self, c0, c1, c2, c3, k):
        """Compute generalized aerodynamic forces matrices
        """
        if k == 0:
            return np.zeros((2, 2), dtype=complex)
        else:
            c_k = 1 - 0.165 / (1 - 0.0455 / k * 1j) - 0.335 / (1 - 0.3 / k * 1j) # Theodorsen function
            q0 = c0[0] * c_k * k + c0[1] * k**2
            q1 = c1[0] * c_k + c1[1] * k + c1[2] * c_k * k + c1[3] * k**2
            q2 = c2[0] * c_k * k + c2[1] * k**2
            q3 = c3[0] * c_k + c3[1] * k + c3[2] * c_k * k + c3[3] * k**2
            return np.array([[q0, q1], [q2, q3]])
