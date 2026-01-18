import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import savemat
import time

def main():
    root = Path(__file__).resolve().parents[1]
    urdf = root / "urdf" / "single_leg_3r_floating.urdf"
    assert urdf.exists(), f"URDF not found: {urdf}"

    # build data from model (fixed base to keep visualization stable)
    model = pin.buildModelFromUrdf(str(urdf))
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
    T    = 10.0
    N    = int(T / dt) + 1
    t    = np.linspace(0.0, T, N)

    # joint trajectory (within limits)
    joint_names = ["hip_pitch", "knee_pitch", "ankle_pitch"]
    joint_ids = [model.getJointId(name) for name in joint_names]
    q_idx = np.array([model.idx_qs[jid] for jid in joint_ids], dtype=int)
    v_idx = np.array([model.idx_vs[jid] for jid in joint_ids], dtype=int)

    q_center = np.array([0.2, -0.4, -0.1])
    q_amp = np.array([0.4, 0.3, 0.2])
    q_phase = np.array([0.0, 1.0, -0.6])
    freq_hz = 0.8
    omega = 2.0 * np.pi * freq_hz

    #  initial state
    q0 = pin.neutral(model)

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
    viz.display(q0)

    # Logs
    com_log = np.zeros((N, 3))
    hg_log  = np.zeros((N, 6))
    hg_err_log = np.zeros((N, 6))
    tau_log = np.zeros((N, model.nv))
    v_log   = np.zeros((N, model.nv))
    q_log   = np.zeros((N, model.nq))

    for k in range(N):
        s = omega * t[k] + q_phase
        qj = q_center + q_amp * np.sin(s)
        vj = q_amp * omega * np.cos(s)
        aj = -q_amp * (omega ** 2) * np.sin(s)

        q = q0.copy()
        v = np.zeros(model.nv)
        a = np.zeros(model.nv)
        q[q_idx] = qj
        v[v_idx] = vj
        a[v_idx] = aj

        if k % 20 == 0:     # 20 * dt = 0.04s which is about 25FPS
            viz.display(q)
            time.sleep(dt*20)

        # CoM and centroidal map/momentum
        pin.centerOfMass(model, data, q, v)
        com = data.com[0].copy()

        pin.ccrba(model, data, q, v)
        Ag = data.Ag.copy()          # 6 x nv
        hg = data.hg.vector.copy()   # 6
        hg_err = hg - Ag @ v

        # torque needed to follow the joint trajectory
        tau = pin.rnea(model, data, q, v, a)

        # Log
        com_log[k, :] = com
        hg_log[k, :]  = hg
        hg_err_log[k, :] = hg_err
        tau_log[k, :] = tau
        v_log[k, :]   = v
        q_log[k, :]   = q

    # Plot: hy and joint states
    hy_log = hg_log[:, 1]
    qj_log = q_log[:, q_idx]
    vj_log = v_log[:, v_idx]
    hg_err_norm = np.linalg.norm(hg_err_log, axis=1)

    print(f"max ||hg - Ag*v||: {hg_err_norm.max():.3e}")

    plt.figure()
    plt.plot(t, hy_log, linewidth=2)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("h_y [N·m·s] (centroidal, world wy)")
    plt.title("Centroidal angular momentum (h_y) from joint motion")

    plt.figure()
    plt.plot(t, qj_log)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("q_j [rad]")
    plt.title("Joint angles (hip, knee, ankle)")
    plt.legend(["hip", "knee", "ankle"])

    plt.figure()
    plt.plot(t, vj_log)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("dq_j [rad/s]")
    plt.title("Joint velocities (hip, knee, ankle)")
    plt.legend(["hip", "knee", "ankle"])

    plt.figure()
    plt.plot(t, hg_err_norm, linewidth=2)
    plt.grid(True)
    plt.xlabel("t [s]")
    plt.ylabel("||hg - Ag*v||")
    plt.title("Centroidal momentum consistency check")

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
        "hg_err": hg_err_log,
    })
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()
