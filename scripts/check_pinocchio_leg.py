import numpy as np
import pinocchio as pin
from pathlib import Path

def main():
    # get urdf from file 
    root = Path(__file__).resolve().parents[1]
    urdf = root / "urdf" / "single_leg_3r_floating.urdf"
    assert urdf.exists(), f"URDF not found: {urdf}"

    # build model from urdf
    model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    data = model.createData()

    # extract neutral configuration and initialize velocity
    q = pin.neutral(model)
    v = np.zeros(model.nv)

    # compute CoM for the floating base which is our joint 0 
    pin.centerOfMass(model, data, q, v)
    com = data.com[0].copy()

    # compute centroidal stuff with pin.ccrba, also has joint jacobians 
    pin.ccrba(model, data, q, v)
    Ag = data.Ag.copy()          # 6 x nv
    hg = data.hg.vector.copy()   # 6 vector: [ang; lin] in WORLD r.f.

    print("=== Model ===")
    print("nq =", model.nq, " nv =", model.nv)
    print("joints:", [model.joints[i].shortname() for i in range(1, model.njoints)])
    print("\n=== CoM (world) ===")
    print("com =", com)

    print("\n=== Centroidal momentum (world) ===")
    print("hg =", hg, "  (order: [wx wy wz vx vy vz])")
 
    # identity check:
    # hg = Ag * v nad and hg = data.Ag
    err = hg - Ag @ v
    print("\n=== Check: hg - Ag*v ===")
    print("err =", err)
    print("||err|| =", np.linalg.norm(err))

if __name__ == "__main__":
    main()

