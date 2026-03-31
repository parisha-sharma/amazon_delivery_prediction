# ============================================================
# ONE-CLICK MODEL TRAINING SCRIPT
# Run this after cloning the project to train all models!
# Command: python train_model.py
# ============================================================

import subprocess
import sys
import os

print("="*55)
print("🚚 Amazon Delivery Time Prediction — Model Trainer")
print("="*55)
print("\nThis script will run the full pipeline automatically!")
print("Please wait — this takes about 3-5 minutes...\n")

scripts = [
    ("notebooks/02_data_cleaning.py",        "🧹 Step 1: Cleaning data..."),
    ("notebooks/03_feature_engineering.py",  "⚙️  Step 2: Engineering features..."),
    ("notebooks/05_model_training.py",        "🤖 Step 3: Training models..."),
]

for script, message in scripts:
    print(message)
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"   ✅ Done!\n")
    else:
        print(f"   ❌ Error in {script}:")
        print(result.stderr)
        sys.exit(1)

print("="*55)
print("🎉 All models trained and saved successfully!")
print("="*55)
print("\n👉 Now run the app with:")
print("   streamlit run app/app.py\n")