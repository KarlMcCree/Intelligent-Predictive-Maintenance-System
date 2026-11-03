#  Edge-AI Predictive Maintenance System

An ESP32-based vibration monitor that predicts machine faults using time-series feature extraction and a Random Forest classifier. This was the foundational project of my edge-AI portfolio, focusing on **Sensing**.

##  Key Results
- **Architecture:** Random Forest (100 trees, max depth 10)
- **Features Extracted:** RMS, Variance, Peak, Crest Factor
- **Accuracy:** > 95% on controlled motor tests
- **Model Size:** < 1 MB (runs entirely in ESP32 SRAM)

##  How it Works
1. An MPU6050 accelerometer is mounted to a DC motor.
2. The ESP32 reads the Z-axis vibration at 100 Hz.
3. Python extracts time-domain features (RMS, Variance, etc.) using a sliding window.
4. A Random Forest is trained to distinguish a "Normal" balanced motor from a "Fault" unbalanced motor.
5. The ESP32 firmware implements the decision logic locally for real-time alerts.

##  How to Run
1. Collect data: `python src/collect_data.py` (Serial interface)
2. Train model: `python -m src.train`
3. Flash firmware: `firmware/edge_inference.ino`

*(See `docs/REPORT.md` for full methodology.)*