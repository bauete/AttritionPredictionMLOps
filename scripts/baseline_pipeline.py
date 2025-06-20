# baseline_pipeline.py
import os
import subprocess
import pandas as pd
import json
import shutil
import time 
import threading
import psutil
import datetime

# Paths to component scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ACQUIRER_SCRIPT = os.path.join(BASE_DIR, "01_data_acquirer.py")
PREPROCESSOR_IBM_SCRIPT = os.path.join(BASE_DIR, "02a_preprocessorIBM.py")
PREPROCESSOR_KPMG_SCRIPT = os.path.join(BASE_DIR, "02b_preprocessorKPMG.py")
FEATURE_ENGINEER_SCRIPT = os.path.join(BASE_DIR, "03_feature_engineer.py")
MODEL_TRAINER_SCRIPT = os.path.join(BASE_DIR, "04_model_trainer.py")
MODEL_EVALUATOR_SCRIPT = os.path.join(BASE_DIR, "05_model_evaluator.py")

# Resource monitoring globals
monitoring_active = False
resource_log_data_list = []

def resource_monitor_thread_function(output_log_path, sampling_interval_sec=1):
    """Thread function to periodically log system-wide CPU and RAM usage."""
    global monitoring_active
    with open(output_log_path, 'w') as f:
        f.write("timestamp,cpu_percent,mem_used_bytes,mem_total_bytes\n")
    while monitoring_active:
        timestamp = pd.Timestamp.now().isoformat()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        with open(output_log_path, 'a') as f:
            f.write(f"{timestamp},{cpu},{mem.used},{mem.total}\n")
        time.sleep(sampling_interval_sec)

def run_script(command, log_file_path):
    """Run a script and log its output."""
    print(f"Running: {' '.join(command)}")
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Executing: {' '.join(command)}\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
        process.wait()
        log_file.write(f"Exit code: {process.returncode}\n\n")
    if process.returncode != 0:
        raise RuntimeError(f"Script failed: {' '.join(command)}")

class StageTimer:
    """Context manager for timing pipeline stages."""
    def __init__(self, stage, metrics_list):
        self.stage = stage
        self.metrics_list = metrics_list
    def __enter__(self):
        self.start = pd.Timestamp.now()
        print(f"\n--- {self.stage} ---")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        end = pd.Timestamp.now()
        duration = (end - self.start).total_seconds()
        self.metrics_list.append({"stage": self.stage, "start": self.start, "end": end, "duration": duration})
        print(f"--- {self.stage} done in {duration:.2f}s ---")

def run_baseline_pipeline(
    source: str, # "ibm", "kpmg" or "dirty_ibm"
    output_dir: str,
    evaluate_only: bool = False,
    reference_artifacts_dir: str = None
):
    """Run the full ML pipeline from raw data to evaluation."""
    global monitoring_active
    pipeline_start = pd.Timestamp.now()
    stage_metrics = []
    os.makedirs(output_dir, exist_ok=True)
    pipeline_log = os.path.join(output_dir, "pipeline_log.txt")
    resource_log = os.path.join(output_dir, "resource_usage.csv")

    # Start resource monitoring
    monitoring_active = True
    monitor_thread = threading.Thread(target=resource_monitor_thread_function, args=(resource_log, 0.2))
    monitor_thread.daemon = True
    monitor_thread.start()
    time.sleep(0.5) # Let monitoring start

    results = {"pipeline_type": "baseline"}
    artifacts_subdir = "artifacts"

    try:
        # --- 1. Data Acquisition ---
        with StageTimer("Data Acquisition", stage_metrics):
            data_path = os.path.join(output_dir, f"01_{source}_acquired_data.csv")
            raw_data_dir = os.path.join(BASE_DIR, "..", "data")
            if source == "ibm":
                input_file = os.path.join(raw_data_dir, "ibm_dataset.csv")
            elif source == "kpmg":
                input_file = os.path.join(raw_data_dir, "kpmg_dataset.csv")
            elif source == "dirty_ibm":
                input_file = os.path.join(raw_data_dir, "ibm_dataset_dirty.csv")
            else:
                raise ValueError(f"Unknown source: {source}")
            run_script(["python", DATA_ACQUIRER_SCRIPT, "--input_file", input_file, "--output_file", data_path], pipeline_log)
            results["acquired_data_path"] = data_path

        # --- 2. Data Preprocessing ---
        with StageTimer("Data Preprocessing", stage_metrics):
            preprocessed_path = os.path.join(output_dir, f"02_{source}_preprocessed_data.csv")
            preprocessor = PREPROCESSOR_IBM_SCRIPT if source in ["ibm", "dirty_ibm"] else PREPROCESSOR_KPMG_SCRIPT
            run_script(["python", preprocessor, "--input_file", data_path, "--output_file", preprocessed_path], pipeline_log)
            results["preprocessed_data_path"] = preprocessed_path

        # --- 3. Feature Engineering ---
        with StageTimer("Feature Engineering", stage_metrics):
            engineered_path = os.path.join(output_dir, f"03_{source}_engineered_features.csv")
            fe_dir = reference_artifacts_dir if evaluate_only and reference_artifacts_dir else output_dir
            os.makedirs(os.path.join(fe_dir, artifacts_subdir), exist_ok=True)
            feature_selector = os.path.join(fe_dir, artifacts_subdir, "feature_selector.pkl")
            expected_cols = os.path.join(fe_dir, artifacts_subdir, "expected_cols.json")
            cmd = [
                "python", FEATURE_ENGINEER_SCRIPT,
                "--input_file", preprocessed_path,
                "--output_file", engineered_path,
                "--feature_selector_path", feature_selector,
                "--expected_cols_path", expected_cols
            ]
            if evaluate_only:
                cmd.append("--apply_only")
            run_script(cmd, pipeline_log)
            results["engineered_data_path"] = engineered_path

        # --- 4. Model Training or Copying ---
        with StageTimer("Model Training", stage_metrics):
            model_dir = os.path.join(output_dir, artifacts_subdir)
            os.makedirs(model_dir, exist_ok=True)
            if evaluate_only:
                if not reference_artifacts_dir:
                    raise ValueError("reference_artifacts_dir required for evaluate_only mode.")
                shutil.copy(os.path.join(reference_artifacts_dir, artifacts_subdir, "model.pkl"), os.path.join(model_dir, "model.pkl"))
                shutil.copy(os.path.join(reference_artifacts_dir, artifacts_subdir, "scaler.pkl"), os.path.join(model_dir, "scaler.pkl"))
                results["model_trained"] = False
                print("Copied reference model and scaler for evaluation.")
            else:
                run_script([
                    "python", MODEL_TRAINER_SCRIPT,
                    "--input_file", engineered_path,
                    "--output_dir", model_dir
                ], pipeline_log)
                results["model_trained"] = True
            results["model_path"] = model_dir

        # --- 5. Model Evaluation ---
        with StageTimer("Model Evaluation", stage_metrics):
            eval_summary = os.path.join(output_dir, "evaluation_summary.json")
            run_script([
                "python", MODEL_EVALUATOR_SCRIPT,
                "--input_file", engineered_path,
                "--model_path", os.path.join(model_dir, "model.pkl"),
                "--output_dir", output_dir
            ], pipeline_log)
            if os.path.exists(eval_summary):
                with open(eval_summary, 'r') as f:
                    results.update(json.load(f))
            results["evaluation_summary_path"] = eval_summary

        print("\nPipeline completed successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        results["error"] = str(e)

    finally:
        # Stop resource monitoring
        monitoring_active = False
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=10)
        # Print timing summary
        print("\n--- Stage Timing ---")
        for m in stage_metrics:
            print(f"{m['stage']}: {m['duration']:.2f}s")
        print("--------------------")
        # Save pipeline log
        with open(pipeline_log, 'a') as f:
            f.write(f"\nPipeline finished at {pd.Timestamp.now().isoformat()}\n")
        print(f"Pipeline log saved to {pipeline_log}")

    return results

if __name__ == "__main__":
    pass