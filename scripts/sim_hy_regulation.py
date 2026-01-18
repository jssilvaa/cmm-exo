import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def main():
    root = Path(__file__).resolve().parents[1]
    urdf = root / "urdf" / "single_leg_3r_floating.urdf"
    assert urdf.exists(), f"URDF not found: {urdf}"

    # build data from model 
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    data  = model.createData()

    # meshcat requires explicit geometry models/data so placements can be updated
    visual_model = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.VISUAL)
    visual_data = pin.GeometryData(visual_model)
    collision_model = pin.buildGeomFromUrdf(model, str(urdf), pin.GeometryType.COLLISION)
    collision_data = pin.GeometryData(collision_model)

    # check model info, namely joint orders
    print("nq:", model.nq, "nv:", model.nv)
    for jid, name in enumerate(model.names):
        print(jid, name)

    #  sim params
    dt   = 0.002
    T    = 3.0
    N    = int(T / dt) + 1
    t    = np.linspace(0.0, T, N)

    #  gains 
    k_h   = 8.0      # centroidal angular momentum damping
    k_dq  = 2.0      # joint velocity damping (regularization)
    reg   = 1e-6     # numerical regularization for least squares

    #  initial state
    q = pin.neutral(model)
    v = np.zeros(model.nv)

    # build visualization 
    viz = MeshcatVisualizer(
        model,
        collision_model,
        visual_model,
        data=data,
        collision_data=collision_data,
        visual_data=visual_data,
    )
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.display(q)

    # velocity kick
    t_kick = 0.5
    dv_kick = np.zeros(model.nv) 
    dv_kick[1] = 0.01 # kick pitch about wy
    kick_i = int(t_kick / dt)

    # Logs
    com_log = np.zeros((N, 3))
    hg_log  = np.zeros((N, 6))
    tau_log = np.zeros((N, model.nv))
    v_log   = np.zeros((N, model.nv))
    q_log   = np.zeros((N, model.nq))

    for k in range(N):
        # kick tick 
        if k == kick_i:
            v = v + dv_kick

        # CoM and centroidal map/momentum
        pin.centerOfMass(model, data, q, v)
        com = data.com[0].copy()

        pin.ccrba(model, data, q, v)
        Ag = data.Ag.copy()          # 6 x nv
        hg = data.hg.vector.copy()   # 6
        M  = data.M.copy()           # nv x nv

        # extract pitch in world frame 
        hy = hg[1]

        # desired hy-dot i.e. pitch angvel
        hdoty_des = -k_h * hy

        # Approximate A_dot row via finite difference
        # probably not gonna work 
        q_next = pin.integrate(model, q, v * dt)
        pin.ccrba(model, data, q_next, v)
        Ag_next = data.Ag.copy()
        Adot = (Ag_next - Ag) / dt

        # row corresponding to wy (angular y)
        a = Ag[1, :].reshape(1, -1)      # 1 x nv
        adotv = (Adot[1, :] @ v)         # scalar

        # solve: a * ddq = hdoty_des - adotv
        rhs = hdoty_des - adotv

        # minimum-norm ddq for a single linear constraint:
        # ddq = a^T * rhs / (a a^T + reg)
        denom = float(a @ a.T) + reg
        ddq_cmd = (a.T.flatten() * (rhs / denom))

        ddq_cmd = ddq_cmd - k_dq * v

        # Convert desired acceleration -> torque (inverse dynamics)
        tau = pin.rnea(model, data, q, v, ddq_cmd)
        # aphysical torque limits 
        tau = clamp(tau, -50.0, 50.0)

        # Forward dynamics (consistent acceleration)
        ddq = pin.aba(model, data, q, v, tau)
        # limit unphysical accelerations
        ddq = clamp(ddq, -50.0, 50.0)

        # Integrate
        # limit unreasonable velocities
        v = v + ddq * dt
        v = clamp(v + ddq * dt, -100.0, 100.0)
        q = pin.integrate(model, q, v * dt)

        # Log
        com_log[k, :] = com
        hg_log[k, :]  = hg
        tau_log[k, :] = tau
        v_log[k, :]   = v
        q_log[k, :]   = q

    # Plot: hy and joint states
    hy_log = hg_log[:, 1]
    qj_log = q_log[:, 7:10] # joint angles
    vj_log = v_log[:, 6:9]  # joint velocities (last 3) 

    plt.figure()
    plt.plot(t, hy_log, linewidth=2)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("h_y [N·m·s] (centroidal, world wy)")
    plt.title("Centroidal angular momentum regulation (h_y → 0)")

    plt.figure()
    plt.plot(t, qj_log)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("q_j [rad]")
    plt.title("Joint angles (ankle, knee, hip)")
    plt.legend(["hip", "knee", "ankle"])

    plt.figure()
    plt.plot(t, vj_log)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("dq_j [rad/s]")
    plt.title("Joint velocities (hip, knee, ankle)")
    plt.legend(["hip", "knee", "ankle"])

    plt.show()

    # Save for MATLAB if we need it later 
    out = root / "logs" / "sim_hy_regulation.mat"
    out.parent.mkdir(parents=True, exist_ok=True) # mkdir if needed
    savemat(str(out), {
        "t": t,
        "q": q_log,
        "dq": v_log,
        "tau": tau_log,
        "com": com_log,
        "hg": hg_log,
    })
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
