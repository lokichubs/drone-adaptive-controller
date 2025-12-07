# MRAC Adaptive controller

# Import libraries
import numpy as np
from base_controller import BaseController
from lqr_solver import dlqr, lqr
from scipy.linalg import solve_continuous_lyapunov, solve_lyapunov, solve_discrete_lyapunov
from math import cos, sin
import numpy as np
from scipy import signal

class AdaptiveController(BaseController):
    """ The LQR controller class.

    """

    def __init__(self, robot, lossOfThurst):
        """ MRAC adaptive controller __init__ method.

        Initialize parameters here.

        Args:
            robot (webots controller object): Controller for the drone.
            lossOfThrust (float): percent lost of thrust.

        """

        super().__init__(robot, lossOfThurst)

        # define integral error
        self.int_e1 = 0
        self.int_e2 = 0
        self.int_e3 = 0
        self.int_e4 = 0

        # flag for initializing adaptive controller
        self.have_initialized_adaptive = False

        # reference model
        self.x_m = None

        # baseline LQR controller gain
        self.Kbl = None

        # Saved matrix for adaptive law computation
        self.A_d = None
        self.B_d = None
        self.Bc_d = None

        self.B = None
        self.Gamma = None
        self.P = None

        # adaptive gain
        self.K_ad = None

    def initializeGainMatrix(self):
        """ Calculate the LQR gain matrix and matrices for adaptive controller.

        """
        n_p = 12 # number of states
        m = 4 # number of integral error terms
        n_u = 4 # number of control inputs

        A_p = np.zeros((n_p, n_p))
        B_p = np.zeros((n_p, n_u))

        A_p[0,6] = 1.0 
        A_p[1,7] = 1.0 
        A_p[2,8] = 1.0
        A_p[3,9] = 1.0
        A_p[4,10] = 1.0
        A_p[5,11] = 1.0
        A_p[6,4] = self.g
        A_p[7,3] = -self.g
        # The remaining four states are input driven (z_ddot, p_dot, q_dot, r_dot)

        B_p[8,0] = 1.0 / self.m
        B_p[9,1] = 1.0 / self.Ix
        B_p[10,2] = 1.0 / self.Iy
        B_p[11,3] = 1.0 / self.Iz


        C_p = np.zeros((m, n_p))
        C_p[0,0] = 1.0     # x
        C_p[1,1] = 1.0     # y
        C_p[2,2] = 1.0     # z
        C_p[3,5] = 1.0     # yaw

        A= np.block([[A_p, np.zeros((n_p, m))],
                      [C_p, np.zeros((m, m))]])
        
        B = np.block([[B_p],
                        [np.zeros((m, n_u))]])
        
        B_c = np.block([[np.zeros((n_p, m))],
                        [-np.eye(m)]])
        
        B_aug = np.block([[B, B_c]])
        
        C= np.block([[C_p, np.zeros((m, m))]])

        D = np.zeros((4, 16))

        
        # Discretize the system using zero-order hold
        A_d, B_aug_d, _, _, _ = signal.cont2discrete((A, B_aug, C, D), self.delT, method='zoh')

        B_d = B_aug_d[:, :n_u]
        Bc_d = B_aug_d[:, n_u:]

        # Record the matrix for later use
        self.B = B # continuous version of B
        self.A_d = A_d  # discrete version of A
        self.B_d = B_d # discrete version of B
        self.Bc_d = Bc_d  # discrete version of Bc

        max_pos = 1
        max_ang = 0.2 * self.pi
        max_vel = 6.0
        max_rate = 0.015 * self.pi
        max_eyI = 3

        max_states = np.array([0.1 * max_pos, 
                               0.1 * max_pos, 
                               1.5*max_pos,
                               2.0*max_ang, 
                               3.0*max_ang, 
                               max_ang,
                               0.5 * max_vel, 
                               0.5 * max_vel, 
                               max_vel,
                               max_rate, 
                               max_rate, 
                               max_rate,
                               0.2 * max_eyI, 
                               0.2 * max_eyI, 
                               1.0 * max_eyI, 
                               0.1 * max_eyI])

        max_inputs = np.array([0.2*self.U1_max, 0.8*self.U1_max, 0.8*self.U1_max, 0.8*self.U1_max])
        Q = np.diag(1/max_states**2)
        R = np.diag(1/(max_inputs)**2)

        # solve for LQR gains   
        [K, _, _] = dlqr(A_d, B_d, Q, R)
        self.Kbl = -K

        [K_CT, _, _] = lqr(A,B, Q, R)
        Kbl_CT = -K_CT

        # initialize adaptive controller gain to baseline LQR controller gain
        self.K_ad = self.Kbl.T

        self.Gamma = 1.0e-3 * np.eye(16)

        Q_lyap = np.copy(Q)
        Q_lyap[0:3,0:3] *= 2
        Q_lyap[2,2]      *= 2            
        Q_lyap[6:9,6:9] *= 2
        Q_lyap[8,8] *= 2
        Q_lyap[14,14] *= 1e-5

        A_m = A + self.B @ Kbl_CT
        self.P = solve_continuous_lyapunov(A_m.T, -Q_lyap)

    def update(self, r):
        """ Get current states and calculate desired control input.

        Args:
            r (np.array): reference trajectory.

        Returns:
            np.array: states. information of the 16 states.
            np.array: U. desired control input.

        """

        U = np.array([0.0, 0.0, 0.0, 0.0]).reshape(-1,1)

        # Fetch the states from the BaseController method
        x_t = super().getStates()

        # update integral term
        self.int_e1 += float((x_t[0]-r[0])*(self.delT))
        self.int_e2 += float((x_t[1]-r[1])*(self.delT))
        self.int_e3 += float((x_t[2]-r[2])*(self.delT))
        self.int_e4 += float((x_t[5]-r[3])*(self.delT))

        # Assemble error-based states into array
        error_state = np.array([self.int_e1, self.int_e2, self.int_e3, self.int_e4]).reshape((-1,1))
        states = np.concatenate((x_t, error_state))

        # initialize adaptive controller
        if self.have_initialized_adaptive == False:
            print("Initialize adaptive controller")
            self.x_m = states
            self.have_initialized_adaptive = True
        else:
            
            e = states - self.x_m
            dK = -self.Gamma @ (states @ (e.T @ self.P @ self.B)) * self.delT
            self.K_ad = self.K_ad + dK
            
            # compute x_m at k+1
            self.x_m = self.A_d @ self.x_m + self.B_d @ self.Kbl @ self.x_m + self.Bc_d @ r 
            # Compute control input
            U = self.K_ad.T @ states

        # calculate control input
        U[0] += self.g * self.m

        # Return all states and calculated control inputs U
        return states, U