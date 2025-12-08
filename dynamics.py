import casadi as ca


def def_vehicle_states():

    X = ca.SX.sym('X')
    Y = ca.SX.sym('Y')
    psi = ca.SX.sym('psi')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    r = ca.SX.sym('r')
    delta = ca.SX.sym('delta')

    return ca.vertcat(X, Y, psi, vx, vy, r, delta)


class Wheels:

    def __init__(self, suffix=''):

        self.suffix = suffix

        self.J = 1.2  # kg*m^2, wheel rotational inertia
        self.r_wheel = 0.3  # m, wheel radius

        self.wheelbase = 2.5  # m, distance between front and rear axles
        self.track_width = 1.5  # m, distance between left and right wheels
        
        self.omega = ca.SX.sym('omega_' + suffix)
        self.T = ca.SX.sym('T_' + suffix)
        self.slip_ratio = ca.SX.sym('slip_ratio_' + suffix)
        self.slip_angle = ca.SX.sym('slip_angle_' + suffix)
        self.dwdt = ca.SX.sym('dwdt_' + suffix, 1)

        self.magic_formula_params = ca.SX.system('magic_formula_params_' + suffix, 4,2)  # [B, C, D, E]
        self.F_x = ca.SX.sym('F_x_' + suffix)
        self.F_y = ca.SX.sym('F_y_' + suffix)

        self.dt = 0.001  # time step for integration

    def cal_wheel_velocity(self, vehicle_states):

        vx = vehicle_states[3]
        vy = vehicle_states[4]
        r = vehicle_states[5]
        yp = self.wheelbase / 2 if self.suffix in ['fl', 'fr'] else -self.wheelbase / 2
        ryp = r * yp
        delta = vehicle_states[6]

        A = ca.SX([[1, 0, -1],
                   [0, 1,  1]])
        
        R = ca.SX([[ca.cos(delta),  ca.sin(delta)],
                   [-ca.sin(delta), ca.cos(delta)]])
        
        f = ca.Function('wheel_velocity_' + self.suffix, [vx, vy, ryp], [R @ (A @ ca.vertcat(vx, vy, ryp))])

        v_w = f(vx, vy, ryp)

        return v_w[0], v_w[1]

    def cal_slip_ratio_and_slip_angle(self, vehicle_states):

        v_wx, v_wy = self.cal_wheel_velocity(vehicle_states)

        larger_value = ca.fmax(ca.fabs(v_wx), ca.fabs(self.omega * self.r_wheel))
        smaller_value = ca.fmin(ca.fabs(v_wx), ca.fabs(self.omega * self.r_wheel))

        self.slip_ratio = (larger_value - smaller_value) / ca.fmax(larger_value, 0.1)
        self.slip_angle = ca.atan2(v_wy, ca.fmax(ca.fabs(v_wx), 0.1))

        return self.slip_ratio, self.slip_angle

    def cal_longitudinal_force(self):
        B = self.magic_formula_params[0,0]
        C = self.magic_formula_params[1,0]
        D = self.magic_formula_params[2,0]
        E = self.magic_formula_params[3,0]

        s = self.slip_ratio

        self.F_x = D * ca.sin(C * ca.atan(B * s - E * (B * s - ca.atan(B * s))))
        return self.F_x
    
    def cal_lateral_force(self):
        B = self.magic_formula_params[0,1]
        C = self.magic_formula_params[1,1]
        D = self.magic_formula_params[2,1]
        E = self.magic_formula_params[3,1]

        alpha = self.slip_angle

        self.F_y = D * ca.sin(C * ca.atan(B * alpha - E * (B * alpha - ca.atan(B * alpha))))
        return self.F_y

    def setup_integrator(self):
        T = ca.SX.sym('T')
        Fx = ca.SX.sym('Fx')
        omega = ca.SX.sym('omega')

        R = 0.3

        dwdt = (T - R*Fx) / self.J
        ode = {'x': omega, 'p': ca.vertcat(T, Fx), 'ode': dwdt}

        self.integrator = ca.integrator(
            f'wheel_integrator_{self.suffix}',
            'rk',
            ode,
            {'tf': self.dt}
        )

    def cal_omega(self):
        result = self.integrator(x0=self.omega, p=ca.vertcat(self.T, self.F_x))
        self.omega = result['xf']
        return self.omega
