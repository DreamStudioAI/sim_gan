import math

TYPICAL_ODE_N_PARAMS = [0.7, 0.25, -0.5 * math.pi, -7.0, 0.1, -15.0 * math.pi / 180.0,
                        30.0, 0.1, 0.0 * math.pi / 180.0, -3.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                        160.0 * math.pi / 180.0]

TYPICAL_ODE_S_PARAMS = [0.2, 0.25, -0.5 * math.pi, -1.0, 0.1, -15.0 * math.pi / 180.0,
                        30.0, 0.1, 0.0 * math.pi / 180.0, -10.0, 0.1, 15.0 * math.pi / 180.0, 0.2, 0.4,
                        160.0 * math.pi / 180.0]

TYPICAL_ODE_F_PARAMS = [0.8, 0.25, -0.5 * math.pi, -10.0, 0.1, -15.0 * math.pi / 180.0,
                        30.0, 0.1, 0.03 * math.pi / 180.0, -10.0, 0.1, 15.0 * math.pi / 180.0, 0.5, 0.2,
                        160.0 * math.pi / 180.0]

TYPICAL_ODE_V_PARAMS = [0.1, 0.6, -0.5 * math.pi,
                        0.0, 0.1, -15.0 * math.pi / 180.0,
                        30.0, 0.1, 0.00 * math.pi / 180.0,
                        -10.0, 0.1, 15.0 * math.pi / 180.0,
                        0.5, 0.2, 160.0 * math.pi / 180.0]

#
# Helper dict:
#
beat_type_to_typical_param = {'N': TYPICAL_ODE_N_PARAMS, 'S': TYPICAL_ODE_S_PARAMS, 'F': TYPICAL_ODE_F_PARAMS,
                              'V': TYPICAL_ODE_V_PARAMS}
