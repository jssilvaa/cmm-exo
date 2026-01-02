# Centroidal Momentum–Based Balance Control for Human–Exoskeleton Systems

This repository implements a minimal, physics-consistent pipeline for studying
balance control of a human–exoskeleton system using centroidal momentum dynamics
and disturbance-canceling control laws.

The current focus is a **planar (sagittal) floating-base model** with holonomic
contact constraints, used as a stepping stone toward a full **SE(3)** formulation.

---

## Core idea

Main thing here is to cancel impulse disturbances on the spatial momentum vector. 
Both linear and angular momentum. Recent research (Beck et. al, Eveld et. al) prove two key points: 
- CoM velocity changes account for about 40% of the loss of balance in humans, under controlled experiments. 
- The faster the counteracting control is, the easier it is to maintain balance. 

In order to implement a disturbance cancellation based approach (already in literature as well somewhere...?) we need to: 
- Establish how we want to reject those disturbances: actuators on the lower body (assume hip, knees, and ankle). Ankle does most of the work, other two shift the possible modes of momentum cancellation of the ankle joint. 
- How we can map arbitrary disturbances to CoM shifts, and perform a disturbance cancelling approach through other joints. 

For these purposes, we propose:  
- A **Centroidal Momentum Matrix (CMM)** based-formulation. Makes the CoM the basis of attention, and in the Orin Goswami 2013 paper you can find the algorithm on how to map generalized coordinates back and forth from the CoM. 
- **Contact-consistent dynamics** through holonomic-constraints. These model the kinematic chain accordingly.  
- **Disturbance rejection / balance control**. This part we still need to decide how to perform. Biomimetic? Reinforcement Learning? 

So: 

The **long-term goal** is to **maintain balance under external disturbances**, while respecting
contact feasibility (unilateral forces, friction limits, center of pressure).

---

## Repository structure

- `models/`  
  Rigid-body kinematics and dynamics (floating base, centroidal quantities,
  contact Jacobians).

- `control/`  
  Balance controllers:
  - LIP / DCM controllers (toy models with CMM)
  - Human-Exo momentum–based controller (real thing) 

- `simulations/`  
  Numerical experiments and validation scenarios.

- `tests/`  
  Unit tests ensuring physical consistency:
  - Mass matrix symmetry and definiteness
  - Centroidal momentum identities
  - Constraint consistency

- `docs/`  
  Theory notes and modeling assumptions.

---

## Modeling assumptions (current stage)

- Sagittal-plane motion (planar SE(2) embedded in SE(3))
- Rigid bodies
- Holonomic flat-foot contact during stance
- No-slip contact (unilateral/frictional limits handled at control level)

These assumptions are **intentional** and will be relaxed incrementally.

Some other notes: 
- The model also works for humanoids (bipeds) and quadripeds. Just make sure to change the mass-inertia matrix, and the contact forces. Change the constraints in the dynamical model as required. Use Craig's or Flyweight's textbooks for references.
---

## Roadmap

1. Validate planar centroidal dynamics
2. Implement LIP-based disturbance rejection
3. Couple centroidal momentum control with contact constraints
4. Extend formulation to full SE(3)
5. Integrate human–exoskeleton interaction model

---

## References

- Orin, Goswami, Lee — *Centroidal dynamics of a humanoid robot*
- Murray, Li, Sastry — *A Mathematical Introduction to Robotic Manipulation*
- Kajita et al. — *Biped walking pattern generation*

---

## Author

José Silva (zzzmiguel)  
MSc Robotics / Control / Embedded Systems
