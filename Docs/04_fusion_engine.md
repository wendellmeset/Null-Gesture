# Fusion Engine: Dempster-Shafer Evidence Theory

## Why Not Simple Voting or Averaging?

Most teams will average classifier probabilities. This has fundamental problems:

1. **Ignorance blindness**: If a classifier can't decide between A and B, it
   assigns 0.5/0.5. Averaging treats this as equal confidence to a classifier
   that's certain about A. In DS, ignorance is `m(Θ)=1` — neutral in combination.

2. **Conflict mishandling**: If the motion detector says "Bye-Bye" (0.9) and the
   posture detector says "T-Arms" (0.9), a softmax average gives ~0.45 each and
   no winner. DS detects high conflict and can reject both — preventing a
   nonsensical output.

3. **Sensor dropout**: If UWB disconnects, the proximity detector can't fire.
   In a softmax average, you'd need to re-weight all other classifiers. In DS,
   the proximity detector contributes `m(Θ)=1` (vacuous), which is neutral.

---

## Dempster-Shafer Theory (Brief)

A **frame of discernment** Θ = {ω₁, ..., ωₙ} is the set of all possible
hypotheses (here: 15 gestures).

A **mass function** m: 2^Θ → [0,1] assigns belief to subsets of Θ:
- m(A) ≥ 0 for all A ⊆ Θ
- m(∅) = 0
- Σ_{A⊆Θ} m(A) = 1

The mass m(Θ) represents **ignorance** — belief that cannot be assigned to any
specific hypothesis.

**Dempster's combination rule** for two mass functions:
```
(m₁ ⊕ m₂)(A) = Σ_{B∩C=A} m₁(B)·m₂(C) / (1 - K)

where K = Σ_{B∩C=∅} m₁(B)·m₂(C)  is the conflict coefficient
```

**Yager's modification** (used here) sends conflict to Θ instead of
normalizing it away:
```
(m₁ ⊕ m₂)(A) = Σ_{B∩C=A} m₁(B)·m₂(C)  for A ≠ Θ, A ≠ ∅
(m₁ ⊕ m₂)(Θ) = Σ_{B∩C=Θ} m₁(B)·m₂(C) + K
(m₁ ⊕ m₂)(∅) = 0
```
Yager's rule is preferred because high conflict should *increase* our ignorance,
not be swept under the normalization rug.

---

## Null-Gesture Fusion Pipeline

### Step 1: Mass Function Construction

Each detector outputs a belief mass vector over its gesture subset. These are
combined into the full 15-gesture frame:

```
For the Motion detector output {bye_bye: 0.7, unknown: 0.3}:
  → m_motion({ByeBye}) = 0.7
  → m_motion(Θ) = 0.3
  → All other singletons = 0

For the Posture detector output {t_arms: 0.6, clapping: 0.2, unknown: 0.2}:
  → m_posture({TArms}) = 0.6
  → m_posture({Clapping}) = 0.2
  → m_posture(Θ) = 0.2
```

**Key**: We only assign mass to **singletons** (specific gestures) and **Θ**.
Never to arbitrary subsets — it's computationally simpler and just as expressive
for our use case.

### Step 2: Sequential Combination

Mass functions are combined pairwise:
```
m_fused = m_motion ⊕ m_posture ⊕ m_doppler ⊕ m_proximity ⊕ m_hand
```

Combination is associative, so order doesn't matter.

### Step 3: Belief and Plausibility

From the fused mass function:
```
Bel(A) = Σ_{B⊆A} m(B)     # lower bound — evidence FOR A
Pl(A)  = Σ_{B∩A≠∅} m(B)  # upper bound — evidence NOT AGAINST A
```

For singleton hypotheses (our case), Bel({ω}) = m({ω}).

### Step 4: Decision Rule

A gesture is **detected** when ALL of:
1. `Bel({gesture}) > 0.6` (sufficient evidence)
2. `Bel({gesture}) > Bel({θ})` (more evidence than ignorance)
3. `K < 0.3` (conflict is manageable — classifiers mostly agree)
4. `Bel({gesture}) is the maximum` (no higher belief in any other gesture)

If no gesture passes: output "no gesture" (neutral state).

---

## Temporal Filtering & State Machine

Raw fused beliefs can jitter frame-to-frame. A temporal state machine applies:

### Hysteresis Thresholds
```
onset_threshold  = 0.6   # belief must rise above this to START a gesture
offset_threshold = 0.3   # belief must fall below this to END a gesture
```

This prevents flickering: once a gesture is triggered, it stays active until
belief drops significantly.

### Gesture State Machine
```
                 onset condition
    IDLE ────────────────────────────► GESTURE_ACTIVE
      ▲                                      │
      │                                      │ offset condition
      │       ┌──────────────────────────────┘
      │       ▼
      │  COOLDOWN ──────────────────────────► IDLE
      │  (300 ms)       cooldown elapsed
      │
      └── (no gesture detected for 100 ms) ──┘
```

### Kalman Filter on Belief Trajectory
A simple 1D Kalman filter smooths each gesture's belief over time:
- State: [belief, belief_velocity]
- Measurement: raw fused belief
- Process noise Q = 0.01, measurement noise R = 0.1

This eliminates high-frequency jitter while preserving genuine transitions.

### Minimum Duration Filter
A gesture must be continuously active for ≥ 200 ms before it's emitted as an
event. Sub-200 ms activations are noise artifacts.

---

## Output Format

```python
@dataclass
class GestureEvent:
    gesture: str           # gesture name
    confidence: float      # fused belief mass (0-1)
    timestamp: float       # detection time (Unix)
    duration: float        # how long the gesture has been active
    contributing_sensors: list[str]  # sensors that provided evidence
```

Events are emitted via a callback or async generator for real-time consumption.

---

## Conflict Resolution Example

Suppose the system observes a waving hand (Bye-Bye):

```
m_motion:     {ByeBye}: 0.75, Θ: 0.25
m_posture:    Θ: 1.0  (no posture detected)
m_doppler:    Θ: 1.0  (no Soli)
m_proximity:  Θ: 1.0  (no Push/Pull)
m_hand:       Θ: 1.0  (no fist/palm gesture)
```

Combined (Yager's rule):
- No conflict (no competing hypotheses), so K = 0
- m_fused({ByeBye}) = 0.75
- m_fused(Θ) = 0.25
- Bel({ByeBye}) = 0.75 > 0.6 ✓ → Bye-Bye detected

Now suppose a simultaneous ambiguous scenario:

```
m_motion:     {ByeBye}: 0.65, Θ: 0.35
m_posture:    {TArms}: 0.60, Θ: 0.40
```

Combined:
- K = m_motion({ByeBye})·m_posture({TArms}) = 0.65·0.60 = 0.39
- m_fused({ByeBye}) = m_motion({ByeBye})·m_posture(Θ) = 0.65·0.40 = 0.26
- m_fused({TArms}) = m_motion(Θ)·m_posture({TArms}) = 0.35·0.60 = 0.21
- m_fused(Θ) = m_motion(Θ)·m_posture(Θ) + K = 0.35·0.40 + 0.39 = 0.53

Neither Bel({ByeBye})=0.26 nor Bel({TArms})=0.21 passes the 0.6 threshold,
AND K=0.39 > 0.3 (high conflict). Result: **reject** — no gesture emitted.
This is correct: the system is genuinely uncertain, and silence is better
than a wrong guess.
