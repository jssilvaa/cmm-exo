import pinocchio as pin 
import numpy as np  

def inspect_model(model: pin.Model): 
    print("nq=", model.nq, " nv=", model.nv)
    print("njoints=", model.njoints, " nframes=", model.nframes)

    # inspect joints 
    for i in range(1, model.njoints): 
        j = model.joints[i]
        print(f"[{i:2d}] name={model.names[i]:20s} nq={j.nq} nv={j.nv} idx_q={j.idx_q} idx_v={j.idx_v}")

    # inspect frames 
    for i, f in enumerate(model.frames): 
        print(f"[{i:2d}] {f.name:25s} parentJoint={f.parentJoint:2d} placement={f.placement}")

def inspect_config(model: pin.Model): 
    q0 = pin.neutral(model)
    v0 = np.zeros(model.nv)
    print("config sanity")
    print("neutral(q) shape: ", q0.shape, "neutral(v) shape: " v0.shape)

    return q0, v0

def confirm_inverse(model: pin.Model): 
    dt = 1e-3 
    q = pin.neutral(model)
    dv = 0.2 * np.random.randn(model.nv) * dt

    q2 = pin.integrate(model, q, dv)
    dv_back = pin.difference(model, q, q2)

    print("||dv - dv_back|| =", np.linalg.norm(dv - dv_back))

# query frame 
def get_frame_pose(model, data, q, frame_name: str): 
    fid = model.getFrameId(frame_name)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMf = data.oMf[fid]
    return oMf.translation.copy(), oMf.rotation.copy()

def check_linear_jacobian(model: pin.Model, data, q, frame_name: str, eps=1e-6):
    fid = model.getFrameId(frame_name)

    # jacobian in WORLD
    J6 = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.WORLD)
    Jv = J6[3:6, :]

    # p(q)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    p0 = data.oMf[fid].translation.copy()

    # random tangent direction 
    d = np.random.randn(model.nv)
    d /= np.linalg.norm(d)

    q1 = pin.integrate(model, q, d*eps)

    pin.forwardKinematics(model, data, q1)
    pin.updateFramePlacements(model, data)
    p1 = data.oMf[fid].translation.copy()

    fd = (p1 - p0)/eps
    lin = Jv @ d 
    return np.linalg.norm(fd - lin), fd, lin

def dynamics_suite(model: pin.Model, data, q, v, a, tau): 
    M = pin.crba(model, data, q)
    M = (M + M.T) / 2.0 # ensure symmetry
    h = pin.nonLinearEffects(model, data, q, v)

    tau_rnea = pin.rnea(model, data, q, v, a)
    tau_Mah = M @ a + h 

    err_rnea = np.linalg.norm(tau_rnea - tau_Mah)

    a_aba = pin.aba(model, data, q, v, tau)
    resid_aba = np.linalg.norm(M @ aba + h - tau)

    return err_rnea, resid_aba, M, h, tau_rnea, a_aba

def check_dynamics(model: pin.Model, data): 
    q = pin.neutral(model)
    v = 0.1 * np.random.randn(model.nv)
    a = 0.1 * np.random.randn(model.nv)
    tau = 0.1 * np.random.randn(model.nv)

    err_rnea, resid_aba, M, h, tau_rnea, a_aba = dynamics_suite(model, data, q, v, a, tau)
    print("||rnea - (Ma + h)|| =", err_rnea)
    print("ABA residual ||M a + h - tau|| =", resid_aba)
    print("min eig(M) =", np.linalg.eigvalsh(M).min())


if __name__ == "main": 
    model = pin.buildModelFromUrdf("path/to.urdf")
    data  = model.createData()
    inspect_model(model)
    q0, v0 = inspect_config(model)


