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

class PlateMapper(om.ExplicitComponent):
    """OpenMDAO design variable mapper
    """
    def setup(self):
        self.add_input('x_f', val=1., desc='fraction of torsion center position along the chord')
        self.add_input('t', val=1., desc='plate thickness')
        self.add_output('dv_geo', val=np.ones(1), desc='geometric design variables')
        self.add_output('dv_struct', val=np.ones(1), desc='structural design variables')

    def setup_partials(self):
        self.declare_partials(of='dv_geo', wrt='x_f', method='exact')
        self.declare_partials(of='dv_struct', wrt='t', method='exact')

    def compute(self, inputs, outputs):
        outputs['dv_geo'][0] = inputs['x_f'][0]
        outputs['dv_struct'][0] = inputs['t'][0]

    def compute_partials(self, inputs, partials):
        partials['dv_geo', 'x_f'][0] = 1.
        partials['dv_struct', 't'][0] = 1.
