# Sensor Strategy: Physics-Based Assignment

## Principle: "Right Sensor for the Right Gesture"

Other teams will connect everything to everything. We explicitly map each gesture
to its **minimal effective sensor set** based on the physics of the movement. This
reduces setup, reduces noise, and improves accuracy.

---

## Physical Setup (Minimal Configuration)

```
                    ┌──────────────────┐
                    │   mmWave Radar   │
                    │  (IWRL6432)      │  ← Tabletop, facing user at ~50 cm
                    │  60 GHz, 120°    │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │  UWB Responder   │  ← Tabletop, next to radar
                    │  (DWM3001CDK)    │
                    └──────────────────┘
                    ┌──────────────────┐
                    │  RFID Reader     │  ← Tabletop antenna pointing at user
                    │  (M7E Hecto)     │
                    └──────────────────┘

           USER
       ┌───────────┐
       │           │
       │  ┌─────┐  │  ← RFID Tag #1 on left wristband
       │  │LEFT │  │
       │  └─────┘  │
       │           │
       │  ┌─────┐  │  ← RFID Tag #2 on right wristband
       │  │RIGHT│  │
       │  └─────┘  │
       │           │
       │  ┌─────┐  │  ← ESP32 + BMI270 IMU + UWB Initiator
       │  │IMU+ │  │     strapped to RIGHT wrist (dominant hand)
       │  │UWB  │  │
       │  └─────┘  │
       │           │
       └───────────┘
```

**Total devices used**: 1 mmWave, 2 UWB boards, 1 RFID reader, 2 RFID tags, 1 IMU.
(The third UWB board is unused — optional for future 3D trilateration.)

**Setup time**: ~30 seconds. Place tabletop cluster, strap wristband, run.

---

## Gesture → Sensor Mapping

### Group A: Inertial (IMU-driven)
These gestures are defined by hand **motion trajectories** — acceleration and
rotation patterns. IMU is the primary sensor because it directly measures
these quantities at 100 Hz with low latency.

| Gesture          | Physical Signature                                 | IMU Channel    |
|------------------|----------------------------------------------------|----------------|
| Clockwise        | Sustained +z angular velocity (gyro_z > 0, ~circle)| gyro_z integral|
| Anti-Clockwise   | Sustained -z angular velocity                      | gyro_z integral|
| Left             | Impulse in +x acceleration (or -x, frame-dep.)    | accel_x peak   |
| Right            | Impulse in -x acceleration                         | accel_x peak   |
| Bye-Bye          | Oscillating gyro_z ~2-4 Hz (waving)                | gyro_z freq.   |
| One Arm Boxing   | Periodic accel magnitude spikes (~1-3 Hz)          | |a| peaks       |
| Two-Arm Boxing   | Alternating accel spikes + RFID hand ID toggle     | |a| + RFID     |

**Secondary**: mmWave centroid position can corroborate Left/Right.

### Group B: Postural (mmWave-driven)
These gestures are defined by **static body configuration** — where the hands/arms
are in 3D space. mmWave point cloud gives direct spatial information. IMU alone
cannot distinguish T-Arms from Raise Arms (both could produce similar
acceleration during the motion but have different steady-state positions).

| Gesture          | Physical Signature                          | mmWave Feature          |
|------------------|---------------------------------------------|-------------------------|
| T-Arms           | Wide horizontal point spread, high y-variance | spread_x, centroid_y    |
| Raise Arms       | Centroid moving upward, points concentrated high | centroid_y velocity     |
| Clapping         | Two distinct blobs → merge → separate       | cluster count, spread   |

**Secondary**: IMU orientation for T-Arms / Raise Arms confirmation.
**RFID**: Confirms both hands detected for T-Arms, Clapping.

### Group C: Micro-Doppler (mmWave velocity)
Soli (rubbing thumb/index finger together) produces a distinctive **micro-Doppler**
signature — tiny periodic velocity modulations at 5-20 Hz from the fingertips.
This is *exactly* what Google's Soli radar chip detects, and our 60 GHz IWRL6432
has sufficient range resolution to capture it.

| Gesture          | Physical Signature                          | mmWave Feature            |
|------------------|---------------------------------------------|---------------------------|
| Soli             | Micro-Doppler at 5-20 Hz, low spatial spread| velocity spectrum peaks   |

**Primary only**: mmWave velocity channel. IMU is stationary during Soli and
provides no useful signal.

### Group D: Proximity (UWB-driven)
Pull and Push are **distance-change** gestures along the radial axis between hand
and table. UWB gives direct, precise (sub-cm) distance measurements without the
drift that plagues IMU double-integration.

| Gesture          | Physical Signature                     | UWB Feature              |
|------------------|----------------------------------------|--------------------------|
| Pull             | Distance decreasing > threshold rate   | d(distance)/dt < -0.3 m/s|
| Push             | Distance increasing > threshold rate   | d(distance)/dt > +0.3 m/s|

**Secondary**: IMU accel_z confirms direction (accel toward body for pull).

### Group E: Hand Articulation (mmWave + IMU)
These are fine hand/finger gestures that modulate either the mmWave point cloud
shape or the IMU orientation.

| Gesture              | Physical Signature                      | Primary Feature              |
|----------------------|-----------------------------------------|------------------------------|
| Open/Close Fist      | Point count modulation (open hand → more points) | mmWave point_count     |
| Palm Up/Down         | ~180° rotation around wrist x-axis      | IMU gyro_x integral          |

**Secondary**: RFID confirms which hand is gesturing.

---

## Sensor Availability & Graceful Degradation

If a sensor is disconnected or errors, the system continues with reduced gesture
coverage:

| Sensors Available      | Gestures Detectable                                      |
|------------------------|----------------------------------------------------------|
| IMU + UWB + mmWave + RFID | All 15 (full)                                         |
| IMU + mmWave + RFID      | All except Pull/Push (UWB missing)                    |
| IMU + mmWave             | Groups A + B + C (motion, posture, Soli)              |
| IMU only                 | Group A only (motion gestures)                         |
| mmWave only              | Groups B + C (posture, Soli)                           |

The Dempster-Shafer fusion engine handles this automatically — missing
classifiers contribute vacuous belief (m(Θ)=1), which is neutral in DS
combination.

---

## Why This Beats "Feed Everything Into One Model"

1. **No irrelevant features**: The NN approach feeds UWB distance into
   Soli classification (irrelevant noise). We don't.
2. **Smaller models**: Each detector is a 1-5 class problem, not 15-class.
3. **Explainable**: If a gesture is misclassified, we know which detector
   failed and why.
4. **Lower sample complexity**: A 15-class deep net needs thousands of labeled
   samples per gesture. Our specialized detectors need tens to hundreds.
5. **Faster inference**: Five small classifiers in parallel vs. one giant
   forward pass. Total compute is lower.
