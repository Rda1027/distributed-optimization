import numpy as np
from std_msgs.msg import Float64MultiArray as MsgFloat


def format_message(id, curr_k, curr_sigma, curr_grad2, curr_z) -> MsgFloat:
    msg = MsgFloat()
    vars_dim = len(curr_z)
    msg.data = [ float(id), float(curr_k), float(vars_dim), *curr_sigma, *curr_grad2, *curr_z ]
    return msg


def unpack_message(msg) -> dict:
    id = int(msg.data[0])
    k = int(msg.data[1])
    vars_dim = int(msg.data[2])
    sigma_est = np.array(msg.data[3 : 3+vars_dim])
    grad2_est = np.array(msg.data[3+vars_dim : 3+2*vars_dim])
    z = np.array(msg.data[3+2*vars_dim : 3+3*vars_dim])

    return {
        "id": id,
        "k": k,
        "sigma_est": sigma_est,
        "grad2_est": grad2_est,
        "z": z,
    }