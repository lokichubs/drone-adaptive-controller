# Import libraries
import numpy as np
from base_controller import BaseController
from lqr_solver import dlqr, lqr
from scipy.linalg import solve_continuous_lyapunov, solve_lyapunov, solve_discrete_lyapunov
from math import cos, sin
import numpy as np
from scipy import signal

class LQRController(BaseController):
    """ The LQR controller class.

    """

    def __init__(self, robot, lossOfThurst=0):
        """ LQR controller __init__ method.

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

        # define K matrix
        self.K = None

    def initializeGainMatrix(self):
        """ Calculate the gain matrix.

        """

        n_p = 12 # number of states
        m = 4 # number of integral error terms
        n_u = 4 # number of control inputs

        # Refer to quadrator_A_B_derivation.pdf for full derivation

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

        A_aug = np.block([[A_p, np.zeros((n_p, m))],
                      [C_p, np.zeros((m, m))]])
        
        B_aug = np.block([[B_p, np.zeros((n_p, m))],
                        [np.zeros((m, n_u)), -np.eye(m)]])
        
        C_aug = np.block([[C_p, np.zeros((m, m))]])

        D = np.zeros((4, 16))

        # Discretize the system using zero-order hold
        A_d, B_d, _, _, _ = signal.cont2discrete((A_aug, B_aug, C_aug, D), self.delT, method='zoh')

        B_d_d = B_d[:, :n_u]
        B_c_d = B_d[:, n_u:]

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
        [K, _, _] = dlqr(A_d, B_d_d, Q, R)

        self.K = -K

    def update(self, r):
        """ Get current states and calculate desired control input.

        Args:
            r (np.array): reference trajectory.

        Returns:
            np.array: states. information of the 16 states.
            np.array: U. desired control input.

        """

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

        # calculate control input
        U = np.matmul(self.K, states)
        U[0] += self.g * self.m

        # Return all states and calculated control inputs U
        return states, U