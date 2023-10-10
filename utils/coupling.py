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

from .structures import Structures
from .aerodynamics import Aerodynamics
from .flutter import Flutter
import openmdao.api as om

class FlutterGroup(om.Group):
    """Generic aerostructural flutter group
    """
    def initialize(self):
         self.options.declare('struct', desc='structural solver', recordable=False)
         self.options.declare('aero', desc='aerodynamic solver', recordable=False)
         self.options.declare('flutter', desc='flutter solver', recordable=False)

    def setup(self):
        self.add_subsystem('struct', Structures(sol=self.options['struct']), promotes=['dv_geo', 'dv_struct', 'm', 'M', 'K'])
        self.add_subsystem('aero', Aerodynamics(sol=self.options['aero']), promotes=['dv_geo', 'Q_re', 'Q_im'])
        self.add_subsystem('flutter', Flutter(sol=self.options['flutter']), promotes=['M', 'K', 'Q_re', 'Q_im', 'f', 'g', 'ks_g'])
