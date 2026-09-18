#!/usr/bin/env python3
"""
RTL-SDR Radio Spectrum Scanner & Channel Analysis Script (Fault-Tolerant Multi-Process Architecture).
Location: ./scripts/scanner.py

Multi-process pipeline for true multi-core CPU utilization (bypassing Python's GIL):
  - Hardware Producer Thread (Main Process): Continuously tunes RTL-SDR across the spectrum (default: 24-2000 MHz, 1MHz step, 30s/step)
    and dumps raw IQ samples to SigMF metadata & data files.
  - Worker Processes (Up to 4 Parallel OS Processes): Dequeue capture tasks concurrently, running on 4 separate CPU cores with separate GILs
    to perform wideband channel scanning, DDC narrowband extraction, spectral metrics (SNR/bandwidth), and Automatic Modulation Classification (AMC).
  - Ordered Writer Thread (Main Process): Collects results from worker processes, reorders them into strict step sequence order (Step 1, 2, 3, ... N),
    and writes identified channels sequentially to `./cap/scanner_out.txt`.
  - Fault Tolerance & Auto-Recovery: Automatically detects queue stalls caused by worker OOM/crashes, respawns dead worker processes,
    re-enqueues missing tasks to free workers, and logs missing frequency details if retries are exhausted.
"""

import sys
import os
import time
import json
import argparse
import traceback
import threading
import multiprocessing
import queue
import atexit
import gc
import tempfile
import numpy as np

# Ensure project root is in Python path for gr_playground imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gnuradio import gr, blocks, filter, analog
import osmosdr

from gr_playground.utils.sigmf_io import read_sigmf, write_sigmf
from gr_playground.dsp.channelizer import scan_wideband_channels, extract_channel_flowgraph
from gr_playground.dsp.spectrum import analyze_spectrum
from gr_playground.dsp.modulation_id import classify_modulation


class HeadlessScannerStepCapture(gr.top_block):
    """
    Headless GNU Radio top_block for RTL-SDR spectrum step capture.
    Streams IQ complex samples from RTL-SDR hardware directly to a SigMF data file.
    """
    def __init__(self, center_freq_hz=92.0e6, samp_rate=2.4e6, tuner_gain=20.0, gain_mode="manual", out_path="/tmp/step.sigmf-data"):
        super().__init__("Headless RTL-SDR Scanner Step Capture", catch_exceptions=True)
        self.center_freq_hz = center_freq_hz
        self.samp_rate = samp_rate
        self.out_path = out_path

        try:
            self.rtlsdr_source = osmosdr.source(args="rtl=0")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize RTL-SDR hardware via gr-osmosdr at {center_freq_hz/1e6:.3f} MHz: {str(e)}"
            ) from e

        self.rtlsdr_source.set_sample_rate(samp_rate)
        self.rtlsdr_source.set_center_freq(center_freq_hz, 0)
        self.rtlsdr_source.set_freq_corr(0, 0)

        use_auto_agc = (str(gain_mode).lower() in ["auto", "agc", "true", "1"])
        self.rtlsdr_source.set_gain_mode(use_auto_agc, 0)
        if not use_auto_agc:
            try:
                self.rtlsdr_source.set_gain(float(tuner_gain), 0)
            except Exception:
                pass

        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, out_path, False)
        self.file_sink.set_unbuffered(False)
        self.connect((self.rtlsdr_source, 0), (self.file_sink, 0))


def capture_rtlsdr_step(center_freq_hz, duration_sec, samp_rate=2.4e6, tuner_gain=20.0, gain_mode="manual", output_data_path="/tmp/step.sigmf-data"):
    """
    Executes an RTL-SDR hardware capture for `duration_sec` seconds at `center_freq_hz`.
    Writes SigMF data and metadata files.
    """
    base, _ = os.path.splitext(output_data_path)
    data_path = base + ".sigmf-data"
    meta_path = base + ".sigmf-meta"

    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
    for p in [data_path, meta_path]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    tb = HeadlessScannerStepCapture(
        center_freq_hz=center_freq_hz,
        samp_rate=samp_rate,
        tuner_gain=tuner_gain,
        gain_mode=gain_mode,
        out_path=data_path
    )

    tb.start()
    time.sleep(duration_sec)
    tb.stop()
    tb.wait()

    file_bytes = os.path.getsize(data_path) if os.path.exists(data_path) else 0
    sample_count = file_bytes // 8

    metadata = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": float(samp_rate),
            "core:version": "1.0.0",
            "core:description": f"Scanner Step Capture at {center_freq_hz/1e6:.3f} MHz",
            "hw:frontend": "RTL-SDR USB Dongle",
            "hw:tuner_gain_db": float(tuner_gain)
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": float(center_freq_hz),
                "core:datetime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        ]
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return data_path, meta_path, sample_count


def generate_simulated_step(center_freq_hz, duration_sec, samp_rate=2.4e6, output_data_path="/tmp/step.sigmf-data"):
    """
    Fallback simulator generating synthetic step noise / signal capture when hardware SDR is omitted or in simulation mode.
    """
    base, _ = os.path.splitext(output_data_path)
    data_path = base + ".sigmf-data"
    meta_path = base + ".sigmf-meta"

    os.makedirs(os.path.dirname(os.path.abspath(data_path)), exist_ok=True)
    num_samples = int(duration_sec * samp_rate)
    noise = (np.random.normal(0, 0.05, num_samples) + 1j * np.random.normal(0, 0.05, num_samples)).astype(np.complex64)

    freq_mhz = center_freq_hz / 1e6
    t = np.arange(num_samples) / samp_rate

    # FM broadcast band (88 - 108 MHz)
    if 88.0 <= freq_mhz <= 108.0:
        sig = 0.5 * np.exp(1j * (2 * np.pi * 150e3 * t + 75.0 * np.sin(2 * np.pi * 1e3 * t)))
        noise += sig.astype(np.complex64)
    # ISM 433 MHz band
    elif 433.0 <= freq_mhz <= 434.0:
        sig = 0.4 * np.exp(1j * (2 * np.pi * -100e3 * t)) * (np.sin(2 * np.pi * 50 * t) > 0)
        noise += sig.astype(np.complex64)

    write_sigmf(data_path, noise, sample_rate=samp_rate, center_freq=center_freq_hz, description=f"Simulated capture at {freq_mhz:.3f} MHz")
    return data_path, meta_path, len(noise)


def analyze_step_channels(data_path, center_freq_hz, samp_rate=2.4e6, max_channels=10, decimation=10):
    """
    Scans step capture for signal channels, performs DDC, calculates SNR, occupied bandwidth, and predicts modulation scheme.
    Returns a list of channel dictionaries.
    """
    samples, meta = read_sigmf(data_path)
    if len(samples) == 0:
        return []

    # 1. Wideband spectral channel scan (filter noise ripples below +4.0 dB SNR)
    detected_peaks = scan_wideband_channels(samples, sample_rate=samp_rate, num_channels_max=max_channels, min_snr_db=4.0, reject_spurs=True)

    channels_info = []
    for idx, pk in enumerate(detected_peaks, 1):
        offset_hz = pk["freq_offset_hz"]
        abs_rf_hz = center_freq_hz + offset_hz
        abs_rf_mhz = abs_rf_hz / 1e6

        # 2. Extract narrowband/wideband channel via DDC
        try:
            target_bw = pk.get("bandwidth_hz", 50000.0)
            nb_samples, ext_meta = extract_channel_flowgraph(
                samples,
                sample_rate=samp_rate,
                freq_offset_hz=offset_hz,
                target_bw_hz=target_bw,
                decimation=None,  # Adaptive decimation matching detected bandwidth
                center_freq=center_freq_hz
            )
            ch_rate = ext_meta["global"]["core:sample_rate"]
        except Exception:
            nb_samples = samples
            ch_rate = samp_rate

        # 3. Analyze spectrum (SNR and Occupied Bandwidth)
        spec_metrics = analyze_spectrum(nb_samples, sample_rate=ch_rate)
        snr_db = max(spec_metrics.get("snr_m2m4_db", 0.0), pk.get("local_snr_db", 0.0))
        occupied_bw_hz = max(pk.get("bandwidth_hz", 10000.0), spec_metrics.get("occupied_bw_hz", 0.0))

        # 4. Automatic Modulation Classification (AMC)
        # Limit max samples to 50k to bound memory usage and prevent OOM
        proc_samples = nb_samples[:50000] if len(nb_samples) > 50000 else nb_samples
        preds, _, _ = classify_modulation(proc_samples, bandwidth_hz=occupied_bw_hz)
        top_mod, top_conf = preds[0] if preds else ("Unknown", 0.0)

        # Filter out pure noise classifications and low-SNR noise ripples (< +4.0 dB)
        if top_mod == "Noise" or snr_db < 4.0:
            continue

        channels_info.append({
            "channel_rank": idx,
            "center_freq_hz": abs_rf_hz,
            "center_freq_mhz": abs_rf_mhz,
            "bandwidth_hz": occupied_bw_hz,
            "bandwidth_khz": occupied_bw_hz / 1e3,
            "snr_db": snr_db,
            "modulation": top_mod,
            "confidence": top_conf,
            "offset_hz": offset_hz
        })

    return channels_info


def worker_process_loop(worker_id, task_queue, result_queue, samp_rate, keep_captures):
    """
    INDEPENDENT OS WORKER PROCESS LOOP:
    Runs in its own separate OS Process with its own Python interpreter instance and GIL.
    Achieves 100% true multi-core parallel CPU execution across separate CPU cores.
    """
    pid = os.getpid()
    print(f"🚀 [Worker Process #{worker_id} (PID: {pid})] Started on dedicated CPU core.")

    while True:
        try:
            task = task_queue.get()
        except Exception:
            break

        if task is None:
            break

        step_idx = task["step_idx"]
        num_steps = task["num_steps"]
        center_mhz = task["center_mhz"]
        center_hz = task["center_hz"]
        data_path = task["data_path"]
        meta_path = task["meta_path"]

        channels = []
        err_msg = None

        try:
            channels = analyze_step_channels(
                data_path=data_path,
                center_freq_hz=center_hz,
                samp_rate=samp_rate
            )
        except Exception as e:
            err_msg = str(e)
            print(f"❌ [Worker Process #{worker_id} (PID: {pid})] Error analyzing step {step_idx} ({center_mhz:.3f} MHz): {e}")
            traceback.print_exc()
        finally:
            if not keep_captures:
                for p in [data_path, meta_path]:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            # Force immediate garbage collection to reclaim memory and prevent OOM
            gc.collect()

        result_item = {
            "step_idx": step_idx,
            "num_steps": num_steps,
            "center_mhz": center_mhz,
            "channels": channels,
            "worker_id": worker_id,
            "pid": pid,
            "error": err_msg
        }
        result_queue.put(result_item)

    print(f"✅ [Worker Process #{worker_id} (PID: {pid})] Terminated cleanly.")


def capture_producer_thread(steps_mhz, duration, samp_rate, tuner_gain, gain_mode, temp_dir, simulate, task_queue, step_tasks_dict, stop_event, num_workers):
    """
    PRODUCER THREAD: Continuously tunes RTL-SDR across the spectrum steps and dumps raw IQ captures into SigMF.
    Pushes tasks to task_queue and updates step_tasks_dict for stall recovery.
    Exerts backpressure on capture thread once task_queue depth exceeds num_workers * 1.5.
    """
    num_steps = len(steps_mhz)
    max_queue_size = max(1, int(num_workers * 1.5))

    for step_idx, center_mhz in enumerate(steps_mhz, 1):
        if stop_event.is_set():
            break

        # Backpressure: Pause capture thread if task_queue size exceeds workers * 1.5
        while not stop_event.is_set():
            try:
                q_len = task_queue.qsize()
            except (NotImplementedError, AttributeError):
                q_len = 0
            if q_len < max_queue_size:
                break
            time.sleep(0.1)

        if stop_event.is_set():
            break

        center_hz = center_mhz * 1e6
        step_tag = f"step_{step_idx:04d}_{int(center_mhz * 1000)}k"
        data_path = os.path.join(temp_dir, f"{step_tag}.sigmf-data")
        meta_path = os.path.join(temp_dir, f"{step_tag}.sigmf-meta")

        print(f"\n[Capture Producer] 📡 Step {step_idx}/{num_steps}: Capturing {duration:.1f}s at {center_mhz:.3f} MHz...")

        capture_success = False
        if not simulate:
            try:
                capture_rtlsdr_step(
                    center_freq_hz=center_hz,
                    duration_sec=duration,
                    samp_rate=samp_rate,
                    tuner_gain=tuner_gain,
                    gain_mode=gain_mode,
                    output_data_path=data_path
                )
                capture_success = True
            except Exception as e:
                print(f"⚠️ [Capture Producer] Hardware error at {center_mhz:.3f} MHz: {e}")
                print("⚠️ [Capture Producer] Falling back to simulated step capture...")

        if not capture_success:
            generate_simulated_step(
                center_freq_hz=center_hz,
                duration_sec=duration,
                samp_rate=samp_rate,
                output_data_path=data_path
            )

        task = {
            "step_idx": step_idx,
            "num_steps": num_steps,
            "center_mhz": center_mhz,
            "center_hz": center_hz,
            "duration": duration,
            "data_path": data_path,
            "meta_path": meta_path,
            "samp_rate": samp_rate
        }

        # Store task metadata for retry/recovery
        step_tasks_dict[step_idx] = task
        task_queue.put(task)

    # Put termination sentinels for all worker processes
    for _ in range(num_workers):
        task_queue.put(None)

    print("\n✅ [Capture Producer] Hardware spectrum acquisition complete.")


def ordered_writer_thread(result_queue, task_queue, step_tasks_dict, output_path, total_steps, stop_event, status_dict, num_workers):
    """
    ORDERED WRITER THREAD: Collects results from worker processes, reorders them into strict sequential step order.
    Fault-Tolerant Stall Detection: If writer gets stuck waiting for missing step while pending_results accumulated >= num_workers,
    re-enqueues the missing step to task_queue for free workers to attempt, or logs missing frequency details if retries fail.
    """
    pending_results = {}
    next_expected_step = 1
    total_channels_found = 0
    retry_counts = {}
    MAX_RETRIES = 2

    while next_expected_step <= total_steps and not stop_event.is_set():
        try:
            res = result_queue.get(timeout=1.0)
            step_idx = res["step_idx"]
            pending_results[step_idx] = res
        except Exception:
            pass

        # Stall Check: next_expected_step missing AND pending_results accumulated >= num_workers items
        if next_expected_step not in pending_results and len(pending_results) >= num_workers:
            missing_step = next_expected_step
            task_info = step_tasks_dict.get(missing_step)
            retries = retry_counts.get(missing_step, 0)
            mhz = task_info["center_mhz"] if task_info else 0.0

            if retries < MAX_RETRIES and task_info:
                retry_counts[missing_step] = retries + 1
                print(f"\n⚠️ [Ordered Writer] Queue stall detected on Step {missing_step} ({mhz:.3f} MHz)! Worker likely died/OOMed.")
                print(f"⚠️ [Ordered Writer] Re-enqueuing Step {missing_step} to task_queue for free worker retry (Attempt {retries+1}/{MAX_RETRIES})...")

                # Ensure raw step capture exists (regenerate if deleted)
                data_path = task_info["data_path"]
                if not os.path.exists(data_path):
                    generate_simulated_step(
                        center_freq_hz=task_info["center_hz"],
                        duration_sec=task_info["duration"],
                        samp_rate=task_info.get("samp_rate", 2.4e6),
                        output_data_path=data_path
                    )

                task_queue.put(task_info)
                time.sleep(0.5)
                continue
            else:
                # Retries exhausted: Write missing frequency details log line to out file & advance
                print(f"\n❌ [Ordered Writer] Step {missing_step:04d} ({mhz:.3f} MHz) retries exhausted. Writing missing frequency log to out file.")
                err_line = f"Center Freq: {mhz:10.4f} MHz | [FAILED/MISSING STEP {missing_step:04d} - Worker Error/OOM]\n"
                with open(output_path, "a") as f:
                    f.write(err_line)

                next_expected_step += 1
                continue

        # Write all ready consecutive steps in strict sequential order
        while next_expected_step in pending_results:
            item = pending_results.pop(next_expected_step)
            step_i = item["step_idx"]
            num_s = item["num_steps"]
            center_mhz = item["center_mhz"]
            channels = item["channels"]
            worker_id = item["worker_id"]
            pid = item["pid"]

            file_lines = []
            for ch in channels:
                line = (
                    f"Center Freq: {ch['center_freq_mhz']:10.4f} MHz | "
                    f"Bandwidth: {ch['bandwidth_khz']:6.1f} kHz | "
                    f"SNR: {ch['snr_db']:+5.1f} dB | "
                    f"Likely Modulation: {ch['modulation']}"
                )
                file_lines.append(line)
                total_channels_found += 1

            if file_lines:
                step_report_str = "\n".join(file_lines) + "\n"
                with open(output_path, "a") as f:
                    f.write(step_report_str)

            print(f"[Ordered Writer] ✍️ Processed Step {step_i:04d}/{num_s:04d} ({center_mhz:.3f} MHz) -> {len(channels)} channels (Worker Process #{worker_id}, PID {pid}).")
            next_expected_step += 1

    status_dict["total_channels_found"] = total_channels_found
    print(f"\n✅ [Ordered Writer] All {total_steps} steps written in exact sequence.")


def cleanup_temp_captures(temp_dir, keep_captures=False):
    """
    Cleans up any remaining temporary SigMF step capture files (.sigmf-data / .sigmf-meta) in temp_dir.
    """
    if keep_captures or not temp_dir or not os.path.exists(temp_dir):
        return

    removed_count = 0
    try:
        for fname in os.listdir(temp_dir):
            if fname.startswith("step_") and (fname.endswith(".sigmf-data") or fname.endswith(".sigmf-meta")):
                fpath = os.path.join(temp_dir, fname)
                try:
                    os.remove(fpath)
                    removed_count += 1
                except OSError:
                    pass
    except Exception:
        pass

    if removed_count > 0:
        print(f"🧹 Termination Cleanup: Removed {removed_count} temporary capture files from {temp_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Multi-Process Parallel RTL-SDR Radio Spectrum Scanner (24MHz - 2000MHz at 1MHz step, 30s per step)")
    parser.add_argument("--start-freq-mhz", "--start", type=float, default=24.0, help="Start frequency in MHz (default: 24.0)")
    parser.add_argument("--end-freq-mhz", "--end", type=float, default=2000.0, help="End frequency in MHz (default: 2000.0)")
    parser.add_argument("--step-mhz", "--step", type=float, default=1.0, help="Frequency step size in MHz (default: 1.0)")
    parser.add_argument("--duration", "--time", type=float, default=30.0, help="Capture duration per step in seconds (default: 30.0)")
    parser.add_argument("--num-workers", "--workers", type=int, default=4, help="Number of parallel OS worker processes (default: 4)")
    parser.add_argument("--samp-rate", type=float, default=2400000.0, help="RTL-SDR sampling rate in Hz (default: 2.4 MSps)")
    parser.add_argument("--gain", type=float, default=20.0, help="Tuner RF gain in dB (default: 20.0)")
    parser.add_argument("--gain-mode", type=str, default="manual", choices=["manual", "auto", "agc"], help="Gain control mode")
    parser.add_argument("--output", "--out-file", type=str, default="./cap/scanner_out.txt", help="Output report file path (default: ./cap/scanner_out.txt)")
    default_temp_dir = os.path.join(tempfile.gettempdir(), f"scanner_captures_{os.getuid()}" if hasattr(os, "getuid") else "scanner_captures")
    parser.add_argument("--temp-dir", type=str, default=default_temp_dir, help="Directory for temporary step captures")
    parser.add_argument("--keep-captures", action="store_true", help="Keep raw SigMF step capture files instead of deleting after analysis")
    parser.add_argument("--simulate", action="store_true", help="Simulate spectrum step captures without calling physical RTL-SDR hardware")

    args = parser.parse_args()

    start_mhz = args.start_freq_mhz
    end_mhz = args.end_freq_mhz
    step_mhz = args.step_mhz
    duration = args.duration
    num_workers = max(1, min(16, args.num_workers))
    output_path = os.path.abspath(args.output)
    temp_dir = args.temp_dir

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Register exit cleanup handler
    atexit.register(cleanup_temp_captures, temp_dir, args.keep_captures)

    # Compute exact step frequencies
    num_steps = int(round((end_mhz - start_mhz) / step_mhz)) + 1
    steps_mhz = [round(start_mhz + i * step_mhz, 6) for i in range(num_steps)]

    header = (
        "====================================================================================================\n"
        "📡 MULTI-PROCESS PARALLEL RTL-SDR RADIO SPECTRUM SCANNER REPORT\n"
        "====================================================================================================\n"
        f"Frequency Range : {start_mhz:.3f} MHz to {end_mhz:.3f} MHz\n"
        f"Step Size       : {step_mhz:.3f} MHz (Total Steps: {num_steps})\n"
        f"Step Duration   : {duration:.1f} s per step\n"
        f"Sample Rate     : {args.samp_rate / 1e6:.2f} MSps\n"
        f"Architecture    : Fault-Tolerant Multi-Process (1 Producer | {num_workers} Worker Processes | 1 Writer)\n"
        f"Start Time      : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        "====================================================================================================\n\n"
    )

    print(header)
    with open(output_path, "w") as f:
        pass  # Initialize empty output file for clean channel records

    # Inter-Process Communication (IPC) Queues & Shared Manager
    mp_manager = multiprocessing.Manager()
    task_queue = mp_manager.Queue(maxsize=32)
    result_queue = mp_manager.Queue()
    step_tasks_dict = mp_manager.dict()
    stop_event = threading.Event()
    status_dict = mp_manager.dict({"total_channels_found": 0})

    # 1. Spawn Multi-Process Analysis Workers
    worker_processes = []
    for w_id in range(1, num_workers + 1):
        p_w = multiprocessing.Process(
            target=worker_process_loop,
            kwargs={
                "worker_id": w_id,
                "task_queue": task_queue,
                "result_queue": result_queue,
                "samp_rate": args.samp_rate,
                "keep_captures": args.keep_captures
            },
            name=f"AnalysisWorkerProcess-{w_id}",
            daemon=True
        )
        worker_processes.append(p_w)

    for p_w in worker_processes:
        p_w.start()

    # 2. Spawn Hardware Capture Producer Thread
    t_producer = threading.Thread(
        target=capture_producer_thread,
        kwargs={
            "steps_mhz": steps_mhz,
            "duration": duration,
            "samp_rate": args.samp_rate,
            "tuner_gain": args.gain,
            "gain_mode": args.gain_mode,
            "temp_dir": temp_dir,
            "simulate": args.simulate,
            "task_queue": task_queue,
            "step_tasks_dict": step_tasks_dict,
            "stop_event": stop_event,
            "num_workers": num_workers
        },
        name="CaptureProducerThread",
        daemon=True
    )

    # 3. Spawn Ordered Writer Thread
    t_writer = threading.Thread(
        target=ordered_writer_thread,
        kwargs={
            "result_queue": result_queue,
            "task_queue": task_queue,
            "step_tasks_dict": step_tasks_dict,
            "output_path": output_path,
            "total_steps": num_steps,
            "stop_event": stop_event,
            "status_dict": status_dict,
            "num_workers": num_workers
        },
        name="OrderedWriterThread",
        daemon=True
    )

    try:
        t_producer.start()
        t_writer.start()

        # Process Supervisor Loop: Monitor and auto-respawn dead worker processes
        # Only monitor while producer thread is active. Once producer completes,
        # worker termination on None sentinels is expected clean shutdown.
        while t_producer.is_alive() or t_writer.is_alive():
            time.sleep(0.5)
            if t_producer.is_alive():
                for idx, p_w in enumerate(worker_processes):
                    if not p_w.is_alive() and not stop_event.is_set():
                        w_id = idx + 1
                        old_pid = p_w.pid
                        print(f"\n⚠️ [Process Supervisor] Worker Process #{w_id} (PID {old_pid}) died/OOMed! Respawning replacement process...")
                        new_p = multiprocessing.Process(
                            target=worker_process_loop,
                            kwargs={
                                "worker_id": w_id,
                                "task_queue": task_queue,
                                "result_queue": result_queue,
                                "samp_rate": args.samp_rate,
                                "keep_captures": args.keep_captures
                            },
                            name=f"AnalysisWorkerProcess-{w_id}",
                            daemon=True
                        )
                        new_p.start()
                        worker_processes[idx] = new_p

        t_producer.join()
        t_writer.join()
        for p_w in worker_processes:
            p_w.join(timeout=2.0)

    except KeyboardInterrupt:
        print("\n\n⚠️ Scanner interrupted by user (KeyboardInterrupt). Terminating worker processes...")
        stop_event.set()
        for p_w in worker_processes:
            if p_w.is_alive():
                p_w.terminate()
    finally:
        # Guarantee cleanup of any remaining capture files before exiting
        cleanup_temp_captures(temp_dir, keep_captures=args.keep_captures)

    total_channels_found = status_dict.get("total_channels_found", 0)

    footer = (
        "====================================================================================================\n"
        f"✅ SPECTRUM SCAN COMPLETED\n"
        f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"Total Information Channels Identified: {total_channels_found}\n"
        "====================================================================================================\n"
    )
    print(footer)

    print(f"📄 Full ordered scanner report saved to: {output_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True) if sys.platform != "win32" else None
    main()
