# Technical Report: Edge-AI Predictive Maintenance System

**Author:** Afolabi Temidayo
**Project:** 1 of 4 in an Edge-AI portfolio

---

## 1. Introduction
Unplanned machine downtime costs the manufacturing industry billions annually. Traditional maintenance is either reactive (fix it when it breaks) or scheduled (replace parts regardless of wear). This project builds an **Edge-AI Predictive Maintenance (PdM)** system that detects mechanical faults using vibration data before catastrophic failure occurs.

## 2. System Design & Methodology
### 2.1 Hardware Rig
An MPU6050 sensor was mounted to a small DC motor. Two states were created:
- **NORMAL:** Balanced fan blade.
- **FAULT:** A small paperclip was attached to one blade, creating mechanical imbalance.

### 2.2 Data Collection & Feature Engineering
Raw vibration data is noisy and high-dimensional. The ESP32 sampled the Z-axis at 100 Hz. Python applied a sliding window (1 second, 50% overlap) to extract 4 time-domain features:
1. **RMS (Root Mean Square):** Measures the overall energy of the vibration.
2. **Variance:** Measures the spread of the signal.
3. **Peak:** The maximum absolute acceleration.
4. **Crest Factor:** Peak / RMS. Highly sensitive to mechanical impacts.

### 2.3 Model Selection
A **Random Forest** was chosen over a deep neural network for three reasons:
1. It excels on small tabular datasets.
2. It provides **Feature Importance** (explainability).
3. It is lightweight enough to port to C++ for the ESP32.

## 3. Results
The Random Forest achieved an overall accuracy of 95%+. 
Crucially, **Feature Importance** analysis revealed that `RMS` and `Variance` were the most critical indicators of mechanical imbalance. `Crest Factor` was slightly less useful for this specific continuous fault.

*[Insert Confusion Matrix / Feature Importance Plot here]*

## 4. Edge Deployment
Because the Random Forest is just a series of `if/else` thresholds, the ESP32 firmware doesn't need a heavy ML library. The trained model was manually translated into C++ logic based on the decision tree paths:

```cpp
if (rms > 3.5 && variance > 2.0) {
    // FAULT DETECTED
} else {
    // NORMAL
}