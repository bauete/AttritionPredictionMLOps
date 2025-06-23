"""
MLOps Pipeline Script

This script orchestrates the entire MLOps pipeline from data acquisition to model evaluation,
"""
import os
import subprocess
import pandas as pd
import json
import shutil
import time
import threading
import psutil

# Paths to component scripts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ACQUIRER_SCRIPT = os.path.join(BASE_DIR, "01_data_acquirer.py")
DRIFT_DETECTOR_SCRIPT = os.path.join(BASE_DIR, "01b_drift_detector.py")
PREPROCESSOR_IBM_SCRIPT = os.path.join(BASE_DIR, "02a_preprocessorIBM.py")
PREPROCESSOR_KPMG_SCRIPT = os.path.join(BASE_DIR, "02b_preprocessorKPMG.py")
FEATURE_ENGINEER_SCRIPT = os.path.join(BASE_DIR, "03_feature_engineer.py")
MODEL_TRAINER_SCRIPT = os.path.join(BASE_DIR, "04_model_trainer.py")
MODEL_EVALUATOR_SCRIPT = os.path.join(BASE_DIR, "05_model_evaluator.py")
PERFORMANCE_TRACKER_SCRIPT = os.path.join(BASE_DIR, "..", "monitoring", "performance_tracker.py")

# Resource monitoring globals
monitoring_active = False
resource_log_data_list = []

def resource_monitor_thread_function(output_log_path, sampling_interval_sec=0.2):
    """Thread function to periodically log system-wide CPU and RAM usage."""
    global monitoring_active
    global resource_log_data_list
    try:
        with open(output_log_path, 'w') as f:
            f.write("timestamp,cpu_percent_system,virtual_memory_used_bytes_system,virtual_memory_total_bytes_system\n")
        while monitoring_active:
            timestamp = pd.Timestamp.now().isoformat()
            cpu_system = psutil.cpu_percent(interval=None)
            mem_system = psutil.virtual_memory()
            log_entry_values = {
                "timestamp": timestamp,
                "cpu_percent_system": cpu_system,
                "virtual_memory_used_bytes_system": mem_system.used,
                "virtual_memory_total_bytes_system": mem_system.total
            }
            resource_log_data_list.append(log_entry_values)
            with open(output_log_path, 'a') as f:
                f.write(f"{timestamp},{cpu_system},{mem_system.used},{mem_system.total}\n")
            time.sleep(sampling_interval_sec)
    except Exception as e:
        print(f"Error in resource monitor thread: {e}")

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

def run_mlops_pipeline(
    source: str, # "ibm", "kpmg" or "dirty_ibm"
    output_dir: str,
    evaluate_only: bool = False,
    detect_drift: bool = False,
    reference_data_path: str = None,
    reference_artifacts_dir: str = None,
    retrain_on_drift: bool = True
):
    """Run the full MLOps pipeline from raw data to evaluation and performance tracking."""
    global monitoring_active
    global resource_log_data_list

    pipeline_start = pd.Timestamp.now()
    resource_log_data_list = []
    stage_metrics = []
    os.makedirs(output_dir, exist_ok=True)
    pipeline_log = os.path.join(output_dir, "pipeline_log.txt")
    resource_log = os.path.join(output_dir, "resource_usage.csv")

    # Start resource monitoring in background
    monitoring_active = True
    monitor_thread = threading.Thread(target=resource_monitor_thread_function, args=(resource_log, 0.2))
    monitor_thread.daemon = True
    monitor_thread.start()
    time.sleep(0.5) # Let monitoring start

    results = {"pipeline_type": "mlops"}
    artifacts_subdir = "artifacts"
    needs_retraining = not evaluate_only

    try:
        # --- 1. Data Acquisition ---
        with StageTimer("Data Acquisition", stage_metrics):
            acquired_data_path = os.path.join(output_dir, f"01_{source}_acquired_data.csv")
            raw_data_dir = os.path.join(BASE_DIR, "..", "data")
            if source == "ibm":
                input_file = os.path.join(raw_data_dir, "ibm_dataset.csv")
            elif source == "kpmg":
                input_file = os.path.join(raw_data_dir, "kpmg_dataset.csv")
            elif source == "dirty_ibm":
                input_file = os.path.join(raw_data_dir, "ibm_dataset_dirty.csv")
            else:
                raise ValueError(f"Unknown source: {source}")
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Raw data input file not found: {input_file}")
            run_script(["python", DATA_ACQUIRER_SCRIPT, "--input_file", input_file, "--output_file", acquired_data_path], pipeline_log)
            results["acquired_data_path"] = acquired_data_path

        # --- 1b. Drift Detection (Optional) ---
        drift_detected_flag = False
        if detect_drift and not evaluate_only:
            with StageTimer("Drift Detection", stage_metrics):
                if not reference_data_path:
                    print("Warning: reference_data_path not provided for drift detection. Skipping drift detection.")
                    results["drift_status"] = "skipped_no_reference_data"
                elif not os.path.exists(reference_data_path):
                    print(f"Warning: reference_data_path {reference_data_path} not found. Skipping drift detection.")
                    results["drift_status"] = "skipped_reference_data_not_found"
                else:
                    drift_report_path = os.path.join(output_dir, "01b_drift_report.html")
                    drift_status_path = os.path.join(output_dir, "01b_drift_status.json")
                    cmd_drift = [
                        "python", DRIFT_DETECTOR_SCRIPT,
                        "--current_data", acquired_data_path,
                        "--output_report", drift_report_path,
                        "--drift_status_file", drift_status_path
                    ]
                    run_script(cmd_drift, pipeline_log)
                    results["drift_report_path"] = drift_report_path
                    results["drift_status_path"] = drift_status_path
                    if os.path.exists(drift_status_path):
                        with open(drift_status_path, 'r') as f:
                            drift_status_data = json.load(f)
                        drift_detected_flag = drift_status_data.get("drift_detected", False)
                        results["drift_status"] = "detected" if drift_detected_flag else "not_detected"
                        results["drift_details"] = drift_status_data
                        print(f"Drift detected: {drift_detected_flag}")
                    else:
                        results["drift_status"] = "error_status_file_not_found"
        elif evaluate_only:
            results["drift_status"] = "skipped_evaluate_only_mode"
        else:
            results["drift_status"] = "skipped_detect_drift_false"

        # Decide if retraining is needed
        if evaluate_only:
            needs_retraining = False
        elif drift_detected_flag and retrain_on_drift:
            needs_retraining = True
            print("Drift detected and retrain_on_drift is True. Proceeding with retraining.")
        elif drift_detected_flag and not retrain_on_drift:
            needs_retraining = False
            print("Drift detected but retrain_on_drift is False. Model will not be retrained. Using reference model if available.")
        else:
            needs_retraining = not evaluate_only
            print(f"No drift detected or drift check skipped. Retraining status based on evaluate_only: {needs_retraining}")

        # --- 2. Data Preprocessing ---
        with StageTimer("Data Preprocessing", stage_metrics):
            preprocessed_path = os.path.join(output_dir, f"02_{source}_preprocessed_data.csv")
            preprocessor = PREPROCESSOR_IBM_SCRIPT if source in ["ibm", "dirty_ibm"] else PREPROCESSOR_KPMG_SCRIPT
            run_script(["python", preprocessor, "--input_file", acquired_data_path, "--output_file", preprocessed_path], pipeline_log)
            results["preprocessed_data_path"] = preprocessed_path

        # --- 3. Feature Engineering ---
        with StageTimer("Feature Engineering", stage_metrics):
            engineered_path = os.path.join(output_dir, f"03_{source}_engineered_features.csv")
            fe_dir = reference_artifacts_dir if not needs_retraining and reference_artifacts_dir else output_dir
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
            if not needs_retraining and reference_artifacts_dir:
                cmd.append("--apply_only")
            run_script(cmd, pipeline_log)
            results["engineered_data_path"] = engineered_path
            results["feature_selector_path_used"] = feature_selector
            results["expected_cols_path_used"] = expected_cols

        # --- 4. Model Training or Copying ---
        with StageTimer("Model Training", stage_metrics):
            model_dir = os.path.join(output_dir, artifacts_subdir)
            os.makedirs(model_dir, exist_ok=True)
            if needs_retraining:
                run_script([
                    "python", MODEL_TRAINER_SCRIPT,
                    "--input_file", engineered_path,
                    "--output_dir", model_dir
                ], pipeline_log)
                results["model_trained_this_run"] = True
            else:
                if not reference_artifacts_dir:
                    raise ValueError("No retraining indicated, but no reference_artifacts_dir provided to get a model from.")
                shutil.copy(os.path.join(reference_artifacts_dir, artifacts_subdir, "model.pkl"), os.path.join(model_dir, "model.pkl"))
                results["model_trained_this_run"] = False
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


        print("\nMLOps Pipeline completed successfully.")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        results["error"] = str(e)

    finally:
        pipeline_end = pd.Timestamp.now()
        pipeline_duration = pipeline_end - pipeline_start
        results["pipeline_duration_seconds"] = pipeline_duration.total_seconds()

        # Stop resource monitoring
        monitoring_active = False
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=10)

        # Summarize resource usage per stage
        if os.path.exists(resource_log) and resource_log_data_list:
            df_resources = pd.read_csv(resource_log)
            if not df_resources.empty:
                df_resources['timestamp'] = pd.to_datetime(df_resources['timestamp'])
                summary_table = []
                for stage_info in stage_metrics:
                    stage_start = stage_info['start']
                    stage_end = stage_info['end']
                    stage_resources = df_resources[(df_resources['timestamp'] >= stage_start) & (df_resources['timestamp'] <= stage_end)]
                    if not stage_resources.empty:
                        peak_cpu = stage_resources['cpu_percent_system'].max()
                        peak_mem_bytes = stage_resources['virtual_memory_used_bytes_system'].max()
                        peak_mem_gb = peak_mem_bytes / (1024**3)
                    else:
                        peak_cpu = "N/A"
                        peak_mem_gb = "N/A"
                    duration = (stage_end - stage_start).total_seconds()
                    summary_table.append({
                        "Stage": stage_info['stage'],
                        "Time (s)": round(duration, 2),
                        "CPU Peak (%)": peak_cpu,
                        "Memory Peak (GB)": round(peak_mem_gb, 2) if isinstance(peak_mem_gb, (int, float)) else "N/A"
                    })
                results['detailed_metrics_per_stage'] = summary_table
                print("\n--- Pipeline Resource Usage by Stage ---")
                print(pd.DataFrame(summary_table).to_string(index=False))
                print("----------------------------------------\n")

        with open(pipeline_log, 'a') as f:
            f.write(f"\nPipeline finished at {pipeline_end.isoformat()}\n")
            f.write(f"Total pipeline duration: {pipeline_duration}\n")
            f.write(f"Resource usage logged to: {resource_log}\n")
        print(f"Pipeline log saved to {pipeline_log}")
        print(f"Total pipeline duration: {pipeline_duration}")
        print(f"Resource usage logged to: {resource_log}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the MLOps pipeline")
    parser.add_argument("--source", required=True, help="Data source ('ibm' or 'kpmg')")
    parser.add_argument("--output_dir", required=True, help="Directory to save pipeline outputs")
    parser.add_argument("--evaluate_only", action="store_true", help="Skip training and evaluate only")
    parser.add_argument("--detect_drift", action="store_true", help="Detect data drift")
    parser.add_argument("--reference_data", help="Path to reference data for drift detection")
    parser.add_argument("--reference_artifacts", help="Directory with reference artifacts")
    parser.add_argument("--no_retrain_on_drift", action="store_true", help="Don't retrain even if drift is detected")
    args = parser.parse_args()
    results = run_mlops_pipeline(
        source=args.source,
        output_dir=args.output_dir,
        evaluate_only=args.evaluate_only,
        detect_drift=args.detect_drift,
        reference_data_path=args.reference_data,
        reference_artifacts_dir=args.reference_artifacts,
        retrain_on_drift=not args.no_retrain_on_drift
    )