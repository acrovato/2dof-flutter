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
import scipy.linalg as spla

class Flutter(om.ExplicitComponent):
    """OpenMDAO flutter solution wrapper
    """
    def initialize(self):
        self.options.declare('sol', desc='flutter solver', recordable=False)

    def setup(self):
        self.sol = self.options['sol']
        self.add_input('M', val=np.ones((2, 2)), desc='mass matrix')
        self.add_input('K', val=np.ones((2, 2)), desc='stiffness matrix')
        self.add_input('Q_re', val=np.ones((self.sol.n_k, 2, 2)), desc='real part of generalized aerodynamic forces matrices')
        self.add_input('Q_im', val=np.ones((self.sol.n_k, 2, 2)), desc='imag part of generalized aerodynamic forces matrices')
        self.add_output('f', val=np.ones((self.sol.n_u, 2)), desc='frequencies')
        self.add_output('g', val=np.ones((self.sol.n_u, 2)), desc='damping ratios')
        self.add_output('ks_g', val=1., desc='aggregated damping ratios')
        self.add_output('f_speed', val=1., desc='flutter speed')
        self.add_output('f_mode', val=1, desc='flutter mode')

    def setup_partials(self):
        self.declare_partials(of='ks_g', wrt=['M', 'K', 'Q_re', 'Q_im'], method='exact')

    def compute(self, inputs, outputs):
        self.sol.set_dvs(inputs['M'], inputs['K'], inputs['Q_re'] + 1j * inputs['Q_im'])
        self.sol.compute()
        self.sol.find_flutter()
        outputs['f'] = self.sol.f
        outputs['g'] = self.sol.g
        outputs['ks_g'] = self.sol.gks
        outputs['f_speed'] = self.sol.f_speed
        outputs['f_mode'] = self.sol.f_mode

    def compute_partials(self, inputs, partials):
        self.sol.compute_gradients()
        partials['ks_g', 'M'][0] = self.sol.gks_m
        partials['ks_g', 'K'][0] = self.sol.gks_k
        partials['ks_g', 'Q_re'][0] = self.sol.gks_qre
        partials['ks_g', 'Q_im'][0] = self.sol.gks_qim

class NIPK():
    """Non-iterative p-k method for flutter solution
    """
    def __init__(self, k_ref, l_ref, rho_inf, u_inf, rho_ks=1e6, vrb=0):
        self._k = k_ref # reference reduced frequencies
        self._l = l_ref # reference length
        self._rho = rho_inf # dynamic pressure
        self._u = u_inf # airspeed test range
        self._rks = rho_ks # KS aggregation parameter for KS
        self._v = vrb # verbosity level
        self.n_k = len(k_ref) # number of reference reduced frequencies
        self.n_u = len(u_inf) # number of airspeed test points

    def set_dvs(self, m, k, q):
        """Set design variables
        """
        self.m = m # mass matrix
        self.k = k # stiffness matrix
        self.q = q # generalized aerodynamic forces matrices at reference reduced frequencies

    def compute(self):
        """Compute frequencies and dampings
        """
        self._pk = np.zeros((self.n_u, 2, 2), dtype=complex) # eigenvalues (at reference reduced frequencies)
        self._vk = np.zeros((self.n_u, 2, 2, 2), dtype=complex) # eigenmodes (at reference reduced frequencies)
        self._ik = np.zeros((self.n_u, 2, 2), dtype=int) # indices of reduced frequencies required for interpolation
        self._ok = np.zeros((self.n_u, 2, 2)) # frequencies (at reference reduced frequencies)
        self._dk = np.zeros((self.n_u, 2, 2)) # difference between eigenvalues and frequencies (at reference reduced frequencies)
        self._p = np.zeros((self.n_u, 2), dtype=complex) # interpolated eigenvalue
        self.f = np.zeros((self.n_u, 2)) # frequency
        self.g = np.zeros((self.n_u, 2)) # damping
        for i in range(self.n_u):
            # Compute eigenvalues at zero airspeed
            if self._u[i] == 0:
                p, _ = self.__eig(self._rho, 0, self.m, self.k, self.q[0,:,:])
                for j in range(2):
                    self.f[i,j] = p[j] / (2 * np.pi)
                continue
            # Compute eigenvalues for each reference reduced frequency
            p_k = np.zeros((self.n_k, 2), dtype=complex)
            v_k = np.zeros((self.n_k, 2, 2), dtype=complex)
            for k in range(self.n_k):
                p_k[k,:], v_k[k,:,:] = self.__eig(self._rho, self._u[i], self.m, self.k, self.q[k,:,:])
            # Match reduced frequency by linear interpolation for each mode
            for j in range(2):
                omega, delta, idx = self.__find_index(self._u[i], p_k[:,j], self._k)
                p_re, p_im = self.__interp(omega[idx], delta[idx], p_k[idx, j])
                # Compute frequency and damping
                self.f[i,j] = p_im / (2 * np.pi)
                self.g[i,j] = p_re / p_im
                # Save intermediate results
                self._ik[i, j, :] = idx
                self._ok[i, j, :] = omega[idx]
                self._dk[i, j, :] = delta[idx]
                self._pk[i, j, :] = p_k[idx, j]
                self._vk[i, j, :, :] = v_k[idx, :, j]
                self._p[i,j] = p_re + 1j * p_im
        # Compute aggregated damping over modes then over dynamic pressures
        self._ksm = np.zeros(self.n_u)
        for i in range(self.n_u):
            self._ksm[i] = self.__ks(self.g[i,:])
        self.gks = self.__ks(self._ksm)

    def compute_gradients(self):
        """Compute the derivatives of the damping aggregations wrt design variables
        """
        z = np.zeros((2, 2))
        zk = np.zeros((self.n_k, 2, 2))
        # Derivative wrt M and K
        self.gks_m = np.zeros(4)
        self.gks_k = np.zeros(4)
        for i in range(z.size):
            dm = np.zeros((2, 2))
            dm[np.unravel_index(i, dm.shape)] = 1.
            self.gks_m[i] = self.__compute_dg(dm, z, zk+1j*zk)
            dk = np.zeros((2, 2))
            dk[np.unravel_index(i, dk.shape)] = 1.
            self.gks_k[i] = self.__compute_dg(z, dk, zk+1j*zk)
        # Derivative wrt Re(Q) and Im(Q)
        self.gks_qre = np.zeros(self.n_k * 4)
        self.gks_qim = np.zeros(self.n_k * 4)
        for i in range(zk.size):
            dq = np.zeros((self.n_k, 2, 2))
            dq[np.unravel_index(i, dq.shape)] = 1.
            self.gks_qre[i] = self.__compute_dg(z, z, dq+1j*zk)
            dq = np.zeros((self.n_k, 2, 2))
            dq[np.unravel_index(i, dq.shape)] = 1.
            self.gks_qim[i] = self.__compute_dg(z, z, zk+1j*dq)

    def find_flutter(self):
        """Find the flutter speed and display it
        """
        found = False
        self.f_speed = self._u[-1]
        self.f_mode = np.nan
        for i in range(len(self._u) - 1):
            if (found):
                break
            for j in range(2):
                if (self.g[i,j] > 0 or self.g[i,j] < 0 and self.g[i+1,j] > 0):
                    if self._v > 0:
                        print(f'NIPK: flutter found for mode {j} near u_inf = {self._u[i]}')
                    self.f_speed = self._u[i]
                    self.f_mode = j
                    found = True
                    break

    def __eig(self, rho, u, m, k, q):
        """Compute the eigenvalues and the eigenmodes
        """
        p, v = spla.eig(k - 0.5 * rho * u**2 * q, -m) # det(p^2*M + K - q_inf*Q) = 0
        p = np.sqrt(p)
        # Sort
        p = np.where(np.imag(p) < 0, -p, p) # change sign of p if imag(p) < 0
        srt = np.imag(p).argsort()
        # Normalize (optional since scipy does it)
        for j in range(2):
            v[:,j] /= spla.norm(v[:,j])
        return p[srt].T, v[:,srt]

    def __find_index(self, u, p, k):
        """Find the indices between which the frequency must be interpolated
        """
        # Compute the difference between the solution and the input
        delta = np.zeros(len(k))
        omega = np.zeros(len(k))
        for i in range(len(k)):
            omega[i] = k[i] * u / self._l
            delta[i] = np.imag(p[i]) - omega[i]
        # Find the indices between which the difference is zero
        for i in range(len(delta) - 1):
            if (delta[i] < 0 and delta[i+1] > 0) or (delta[i] > 0 and delta[i+1] < 0):
                idx = [i, i+1]
                break
            if i == len(delta) - 2:
                ai = np.argmin(np.abs(delta))
                if ai == 0:
                    idx = [ai, ai+1]
                elif ai == len(delta) - 1:
                    idx = [ai-1, ai]
                else:
                    raise RuntimeError(f'NIPK: could not match frequency for mode {i} at velocity {u}!\n')
        return omega, delta, idx

    def __interp(self, omega, delta, p_k):
        """Match the frequency and interpolate the eigenvalues and eigenvectors
        """
        # Interpolate the frequency at which the difference is zero
        slope = (delta[1] - delta[0]) / (omega[1] - omega[0])
        p_im = (slope * omega[0] - delta[0]) / slope
        # Interpolate the damping
        slope = (np.real(p_k[1]) - np.real(p_k[0])) / (omega[1] - omega[0])
        p_re = np.real(p_k[0]) + slope * (p_im - omega[0])
        return p_re, p_im

    def __ks(self, v):
        """Aggregate the values in a vector using the Kreisselmeier–Steinhauser (KS) method
        """
        v_max = max(v)
        sum = 0
        for i in range(len(v)):
            sum += np.exp(self._rks * (v[i] - v_max))
        return v_max + 1/self._rks * np.log(sum)

    def __compute_dg(self, dm, dk, dq_k):
        """Compute the derivative of the damping aggregation wrt one input
        """
        dg = np.zeros((self.n_u, 2))
        for i in range(self.n_u):
            # Derivative of eigenvalue solution at zero airspeed
            if self._u[i] == 0:
                continue
            # Derivative of eigenvalue solution
            dp_k = np.zeros((2, 2), dtype=complex)
            for j in range(2):
                for k in range(2):
                    q = self.q[self._ik[i,j,k],:,:]
                    p_k = self._pk[i,j,k]
                    v_k = self._vk[i,j,k,:].T
                    dq = dq_k[self._ik[i,j,k],:,:]
                    dp_k[j,k] = self.__deig(self._rho, self._u[i], self.m, self.k, q, p_k, v_k, dm, dk, dq)
            for j in range(2):
                # Derivative of eigenvalue interpolation
                dp_re, dp_im = self.__dinterp(self._ok[i,j,:], self._dk[i,j,:], self._pk[i,j,:], np.imag(self._p[i,j]), dp_k[j,:])
                # Derivative of damping
                dg[i,j] = dp_re / np.imag(self._p[i,j]) - np.real(self._p[i,j]) / np.imag(self._p[i,j])**2 * dp_im
        # Derivative of double KS aggregation
        dksm = np.zeros(self.n_u)
        for i in range(self.n_u):
            dksm[i] = self.__dks(self.g[i,:], dg[i,:])
        return self.__dks(self._ksm, dksm)

    def __deig(self, rho, u, m, k, q, p_k, v_k, dm, dk, dq):
        """Compute the derivative of an eigenvalue problem
        """
        # Components
        a0 = -2 * p_k * m @ v_k
        a1 = p_k**2 * m + k - 0.5 * rho * u**2 * q
        a2 = 0
        a3 = v_k.T
        b0 = (p_k**2 * dm + dk - 0.5 * rho * u**2 * dq) @ v_k
        b1 = 0
        # SoE
        a = [[a0[0], a1[0,0], a1[0,1]], [a0[1], a1[1,0], a1[1,1]], [a2, a3[0], a3[1]]]
        b = [[b0[0]], [b0[1]], [b1]]
        x = spla.lu_solve((spla.lu_factor(a)), b)
        return x[0,0]

    def __dinterp(self, omega, delta, p_k, p_im, dp_k):
        """Compute the derivative of a linear interpolation
        """
        # x: 0 = s * (x - x0) + y0
        slope = (delta[1] - delta[0]) / (omega[1] - omega[0])
        d_slope = (np.imag(dp_k[1]) - np.imag(dp_k[0])) / (omega[1] - omega[0])
        dp_im = (d_slope * omega[0] - np.imag(dp_k[0])) / slope - (slope * omega[0] - delta[0]) / slope**2 * d_slope
        # z: z = s * (x - x0) + z0
        slope = (np.real(p_k[1]) - np.real(p_k[0])) / (omega[1] - omega[0])
        d_slope = (np.real(dp_k[1]) - np.real(dp_k[0])) / (omega[1] - omega[0])
        dp_re = np.real(dp_k[0]) + d_slope * (p_im - omega[0]) + slope * dp_im
        return dp_re, dp_im

    def __dks(self, v, dv):
        """Compute the derivative of a KS aggregation
        """
        v_max = max(v)
        sum = [0, 0]
        for i in range(len(v)):
            sum[0] += np.exp(self._rks * (v[i] - v_max)) * dv[i]
            sum[1] += np.exp(self._rks * (v[i] - v_max))
        return sum[0] / sum[1]
