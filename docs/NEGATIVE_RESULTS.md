# Negative and blocked results

| Attempt | Original question / implementation | Result and reason | Lesson / revisit |
|---|---|---|---|
| edge-bowl | Can a displaced target create a recoverable failure? | Moving the bowl made the policy OOD and it did not contact the hazard. | Keep hazard on the nominal swept path; do not use as collision evidence. |
| kitchen fixture | Can a second task establish the same effect? | Nominal competence was too low to isolate a collision mechanism. | Revisit only after a nominal bridge gate on a suitable task. |
| cookies box | Can a short object obstruct the reach? | Object was too low to enter the swept volume. | Geometry must be validated against full-arm swept volume. |
| grasp snapshot | Can active grasp force be reproduced by state reset? | State-reset round-trip did not preserve the active grasp state. | Treat as simulator-state limitation; do not claim a grasp recovery witness. |
| final-readout steering | Can probe-direction steering induce braking? | All tested alphas failed to reduce crashes; large alpha did not produce braking. | Negative mechanism evidence; try a different intervention only with a separate hypothesis. |
| d70/d78/d85 detour | Can OSC end-effector detours complete the task around tall walls? | Full-arm geometry, especially link5/elbow, collides with tall walls. | Tall-wall task completion remains unsolved. |
| wall-to-glass probe | Does a wall-trained probe transfer to glass? | It does not transfer across hazards. | Hazard-specific representations/operating modes are a scope boundary. |
| gradient/BORDER wall mode | Does the current wall probe cover transition-band crashes? | Current evidence indicates it may miss this crash mode. | Do not generalize guard coverage beyond the on-path wall mode. |
| P0 two-task held-out dissociation/guard | Does hidden state beat all preregistered baselines and improve held-out online safety? | Calibration and held-out capture had zero positive T-5 frames; held-out vanilla had 0/50 crashes. AUC differences, crash reduction, and counterfactual lead time are not identifiable; `dissociation_supported=false`. | Accept as negative/indeterminate. Do not rescue by changing held-out scenarios, horizon, seeds, repeats, or thresholds; any follow-up is a new preregistered study. |
