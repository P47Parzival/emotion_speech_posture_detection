# Emotion Detection Model - Context

There are **two available files** for training the emotion detection model:

1. **Emotion_detection_best_accuracy_code**
   - Uses **3 algorithms**: SVM, RG, and GB.
   - Provides **higher accuracy** but takes a **very long time** (around a day) to train.
   - Generates:
     - `ensemble_model.pkl`
     - `scaler_ensemble_model.pkl`

2. **Emotion_detection_half_accuracy_code**
   - **Significantly faster** (training time: ~2–3 minutes).
   - Provides **around 51% accuracy**.
   - Generates:
     - `ensemble_model_new.pkl`
     - `scaler_new.pkl`

---

### Usage Note
- Choice of code depends on your priority: **accuracy vs. speed**.
- For further execution:
  - Use **`ensemble_model.pkl` with `scaler_ensemble_model.pkl`** if running the best accuracy code.
  - Use **`ensemble_model_new.pkl` with `scaler_new.pkl`** if running the half accuracy code.

- Make sure to use your pc at best performance mode not on balanced or power efficiency mode to get faster results, rest the choice is yours.