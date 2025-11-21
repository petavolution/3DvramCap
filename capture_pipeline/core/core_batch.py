#!/usr/bin/env python3
"""
Batch Processing with Human Supervision
========================================
Automated batch processing with checkpoints for human review.

Features:
    - Process multiple RenderDoc captures
    - Checkpoint system for human approval
    - Progress tracking and reporting
    - Failure recovery and retry
    - Parallel processing support
    - Email/webhook notifications (optional)

Usage:
    from core_batch import BatchProcessor, BatchConfig

    processor = BatchProcessor(config)
    processor.add_captures(["scene1.rdc", "scene2.rdc"])
    processor.run(checkpoint_callback=my_review_function)

Author: Capture Pipeline
"""

import json
import time
import hashlib
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import shutil


class TaskStatus(Enum):
    """Status of a batch task."""
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class CheckpointType(Enum):
    """Types of checkpoints requiring human review."""
    EXTRACTION_COMPLETE = "extraction_complete"
    DEDUPLICATION_COMPLETE = "deduplication_complete"
    MESH_REVIEW = "mesh_review"
    TEXTURE_REVIEW = "texture_review"
    PRE_EXPORT = "pre_export"
    EXPORT_COMPLETE = "export_complete"
    QUALITY_CHECK = "quality_check"


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    status: TaskStatus
    start_time: float
    end_time: float = 0
    output_files: List[str] = field(default_factory=list)
    mesh_count: int = 0
    texture_count: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def to_dict(self) -> dict:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'duration_seconds': self.duration,
            'output_files': self.output_files,
            'mesh_count': self.mesh_count,
            'texture_count': self.texture_count,
            'errors': self.errors,
            'warnings': self.warnings,
            'metrics': self.metrics
        }


@dataclass
class Checkpoint:
    """A checkpoint requiring human review."""
    checkpoint_type: CheckpointType
    task_id: str
    timestamp: float
    data: Dict[str, Any]
    preview_files: List[str] = field(default_factory=list)
    auto_approve_after: Optional[float] = None  # seconds, None = manual only
    approved: Optional[bool] = None
    reviewer_notes: str = ""

    def to_dict(self) -> dict:
        return {
            'type': self.checkpoint_type.value,
            'task_id': self.task_id,
            'timestamp': datetime.fromtimestamp(self.timestamp).isoformat(),
            'data': self.data,
            'preview_files': self.preview_files,
            'auto_approve_after': self.auto_approve_after,
            'approved': self.approved,
            'notes': self.reviewer_notes
        }


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    # Paths
    input_dir: str = "captures"
    output_dir: str = "export"
    work_dir: str = "work"
    log_dir: str = "logs"

    # Tools
    renderdoc_cmd: str = "renderdoccmd"
    blender_path: str = "blender"
    python_path: str = "python"

    # Processing
    max_parallel: int = 1  # Parallel captures (1 = sequential)
    max_meshes_per_capture: int = 500
    min_vertices: int = 50
    scale_factor: float = 0.01

    # Checkpoints
    checkpoint_after_extract: bool = True
    checkpoint_after_dedup: bool = False
    checkpoint_before_export: bool = True
    checkpoint_after_export: bool = True
    auto_approve_timeout: Optional[float] = None  # Auto-approve after N seconds

    # Export
    export_gltf: bool = True
    export_usd: bool = True
    export_blend: bool = False
    use_unlit: bool = True

    # Notifications (optional)
    webhook_url: Optional[str] = None
    email_on_complete: Optional[str] = None

    # Recovery
    resume_from_checkpoint: bool = True
    max_retries: int = 2

    def to_dict(self) -> dict:
        return {
            'paths': {
                'input': self.input_dir,
                'output': self.output_dir,
                'work': self.work_dir,
                'logs': self.log_dir
            },
            'tools': {
                'renderdoc': self.renderdoc_cmd,
                'blender': self.blender_path,
                'python': self.python_path
            },
            'processing': {
                'max_parallel': self.max_parallel,
                'max_meshes': self.max_meshes_per_capture,
                'min_vertices': self.min_vertices,
                'scale_factor': self.scale_factor
            },
            'checkpoints': {
                'after_extract': self.checkpoint_after_extract,
                'after_dedup': self.checkpoint_after_dedup,
                'before_export': self.checkpoint_before_export,
                'after_export': self.checkpoint_after_export,
                'auto_approve_timeout': self.auto_approve_timeout
            },
            'export': {
                'gltf': self.export_gltf,
                'usd': self.export_usd,
                'blend': self.export_blend,
                'unlit': self.use_unlit
            }
        }

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'BatchConfig':
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = cls()
        if 'paths' in data:
            config.input_dir = data['paths'].get('input', config.input_dir)
            config.output_dir = data['paths'].get('output', config.output_dir)
            config.work_dir = data['paths'].get('work', config.work_dir)
            config.log_dir = data['paths'].get('logs', config.log_dir)

        if 'processing' in data:
            config.max_parallel = data['processing'].get('max_parallel', 1)
            config.max_meshes_per_capture = data['processing'].get('max_meshes', 500)
            config.min_vertices = data['processing'].get('min_vertices', 50)

        if 'checkpoints' in data:
            config.checkpoint_after_extract = data['checkpoints'].get('after_extract', True)
            config.checkpoint_after_dedup = data['checkpoints'].get('after_dedup', False)
            config.checkpoint_before_export = data['checkpoints'].get('before_export', True)
            config.auto_approve_timeout = data['checkpoints'].get('auto_approve_timeout')

        if 'export' in data:
            config.export_gltf = data['export'].get('gltf', True)
            config.export_usd = data['export'].get('usd', True)
            config.export_blend = data['export'].get('blend', False)

        return config


class BatchProcessor:
    """
    Batch processor with human supervision checkpoints.

    Usage:
        processor = BatchProcessor(config)
        processor.add_captures(["scene1.rdc", "scene2.rdc"])

        # With interactive review
        processor.run(checkpoint_callback=interactive_review)

        # Or auto-approve all
        processor.run(auto_approve=True)
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        self.config = config or BatchConfig()
        self.tasks: Dict[str, Dict] = {}
        self.results: Dict[str, TaskResult] = {}
        self.checkpoints: List[Checkpoint] = []
        self.current_checkpoint: Optional[Checkpoint] = None

        # State file for recovery
        self.state_file = Path(self.config.work_dir) / "batch_state.json"

        # Ensure directories exist
        for dir_path in [self.config.work_dir, self.config.log_dir, self.config.output_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def add_capture(self, capture_path: str, task_id: Optional[str] = None) -> str:
        """Add a single capture to the batch."""
        path = Path(capture_path)
        if not path.exists():
            raise FileNotFoundError(f"Capture not found: {capture_path}")

        if task_id is None:
            task_id = path.stem

        # Make task_id unique if duplicate
        base_id = task_id
        counter = 1
        while task_id in self.tasks:
            task_id = f"{base_id}_{counter}"
            counter += 1

        self.tasks[task_id] = {
            'capture_path': str(path.absolute()),
            'status': TaskStatus.PENDING.value,
            'added_at': time.time()
        }

        return task_id

    def add_captures(self, capture_paths: List[str]) -> List[str]:
        """Add multiple captures to the batch."""
        return [self.add_capture(p) for p in capture_paths]

    def add_captures_from_directory(self, directory: str, pattern: str = "*.rdc") -> List[str]:
        """Add all captures matching pattern from directory."""
        import glob
        captures = glob.glob(str(Path(directory) / pattern))
        return self.add_captures(sorted(captures))

    def _save_state(self):
        """Save current state for recovery."""
        state = {
            'tasks': self.tasks,
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'timestamp': time.time()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> bool:
        """Load state from file if exists."""
        if not self.state_file.exists():
            return False

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            self.tasks = state.get('tasks', {})
            # Results would need proper reconstruction
            print(f"Loaded state with {len(self.tasks)} tasks")
            return True
        except Exception as e:
            print(f"Failed to load state: {e}")
            return False

    def _create_checkpoint(self,
                          checkpoint_type: CheckpointType,
                          task_id: str,
                          data: Dict[str, Any],
                          preview_files: Optional[List[str]] = None) -> Checkpoint:
        """Create a checkpoint for human review."""
        checkpoint = Checkpoint(
            checkpoint_type=checkpoint_type,
            task_id=task_id,
            timestamp=time.time(),
            data=data,
            preview_files=preview_files or [],
            auto_approve_after=self.config.auto_approve_timeout
        )

        self.checkpoints.append(checkpoint)
        self.current_checkpoint = checkpoint
        self._save_state()

        return checkpoint

    def _wait_for_approval(self,
                          checkpoint: Checkpoint,
                          callback: Optional[Callable[[Checkpoint], bool]] = None,
                          auto_approve: bool = False) -> bool:
        """Wait for human approval of checkpoint."""
        if auto_approve:
            checkpoint.approved = True
            checkpoint.reviewer_notes = "Auto-approved"
            return True

        if callback:
            # Call user-provided review function
            result = callback(checkpoint)
            checkpoint.approved = result
            return result

        # Auto-approve after timeout if configured
        if checkpoint.auto_approve_after:
            print(f"\n[CHECKPOINT] {checkpoint.checkpoint_type.value}")
            print(f"Task: {checkpoint.task_id}")
            print(f"Auto-approving in {checkpoint.auto_approve_after} seconds...")
            print("Press Ctrl+C to review manually")

            try:
                time.sleep(checkpoint.auto_approve_after)
                checkpoint.approved = True
                checkpoint.reviewer_notes = "Auto-approved after timeout"
                return True
            except KeyboardInterrupt:
                pass

        # Interactive approval
        return self._interactive_approval(checkpoint)

    def _interactive_approval(self, checkpoint: Checkpoint) -> bool:
        """Interactive command-line approval."""
        print("\n" + "=" * 60)
        print(f"CHECKPOINT: {checkpoint.checkpoint_type.value}")
        print("=" * 60)
        print(f"Task ID: {checkpoint.task_id}")
        print(f"Time: {datetime.fromtimestamp(checkpoint.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Show data summary
        for key, value in checkpoint.data.items():
            if isinstance(value, (int, float, str)):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) <= 5:
                print(f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: [{len(value)} items]")

        # Show preview files
        if checkpoint.preview_files:
            print("\nPreview files:")
            for pf in checkpoint.preview_files[:5]:
                print(f"  - {pf}")
            if len(checkpoint.preview_files) > 5:
                print(f"  ... and {len(checkpoint.preview_files) - 5} more")

        print()
        while True:
            response = input("[A]pprove / [R]eject / [S]kip / [V]iew files? ").strip().lower()

            if response in ('a', 'approve', 'y', 'yes'):
                notes = input("Notes (optional): ").strip()
                checkpoint.approved = True
                checkpoint.reviewer_notes = notes
                return True

            elif response in ('r', 'reject', 'n', 'no'):
                notes = input("Rejection reason: ").strip()
                checkpoint.approved = False
                checkpoint.reviewer_notes = notes
                return False

            elif response in ('s', 'skip'):
                checkpoint.approved = None
                checkpoint.reviewer_notes = "Skipped"
                return False

            elif response in ('v', 'view'):
                if checkpoint.preview_files:
                    print("\nFiles to review:")
                    for i, pf in enumerate(checkpoint.preview_files):
                        print(f"  [{i+1}] {pf}")
                else:
                    print("No preview files available")

            else:
                print("Invalid response. Enter A/R/S/V")

    def _run_extraction(self, task_id: str, capture_path: str) -> TaskResult:
        """Run RenderDoc extraction for a capture."""
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )

        # Output directory for this task
        task_output = Path(self.config.work_dir) / task_id / "extracted"
        task_output.mkdir(parents=True, exist_ok=True)

        # Build extraction command
        extract_script = Path(__file__).parent / "core_extract.py"

        cmd = [
            self.config.renderdoc_cmd, "python",
            str(extract_script),
            capture_path,
            str(task_output),
            "--max-meshes", str(self.config.max_meshes_per_capture),
            "--min-vertices", str(self.config.min_vertices)
        ]

        try:
            log_file = Path(self.config.log_dir) / f"{task_id}_extract.log"

            with open(log_file, 'w') as log:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600  # 10 minute timeout
                )

            if proc.returncode == 0:
                # Count outputs
                obj_files = list(task_output.glob("*.obj"))
                png_files = list(task_output.glob("*.png"))

                result.output_files = [str(f) for f in obj_files]
                result.mesh_count = len(obj_files)
                result.texture_count = len(png_files)
                result.status = TaskStatus.COMPLETED

                # Load stats if available
                stats_file = task_output / "extraction_stats.json"
                if stats_file.exists():
                    with open(stats_file) as f:
                        result.metrics = json.load(f)
            else:
                result.status = TaskStatus.FAILED
                result.errors.append(f"Extraction failed with code {proc.returncode}")

        except subprocess.TimeoutExpired:
            result.status = TaskStatus.FAILED
            result.errors.append("Extraction timed out after 10 minutes")
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.errors.append(str(e))

        result.end_time = time.time()
        return result

    def _run_deduplication(self, task_id: str) -> TaskResult:
        """Run mesh deduplication."""
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )

        input_dir = Path(self.config.work_dir) / task_id / "extracted"
        output_dir = Path(self.config.work_dir) / task_id / "deduplicated"
        output_dir.mkdir(parents=True, exist_ok=True)

        process_script = Path(__file__).parent / "core_process.py"

        cmd = [
            self.config.python_path,
            str(process_script),
            str(input_dir),
            str(output_dir),
            "--min-vertices", str(self.config.min_vertices)
        ]

        try:
            log_file = Path(self.config.log_dir) / f"{task_id}_dedup.log"

            with open(log_file, 'w') as log:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=300
                )

            if proc.returncode == 0:
                obj_files = list(output_dir.glob("*.obj"))
                result.output_files = [str(f) for f in obj_files]
                result.mesh_count = len(obj_files)
                result.status = TaskStatus.COMPLETED

                # Calculate dedup stats
                input_count = len(list(input_dir.glob("*.obj")))
                result.metrics['input_meshes'] = input_count
                result.metrics['output_meshes'] = len(obj_files)
                result.metrics['duplicates_removed'] = input_count - len(obj_files)
            else:
                result.status = TaskStatus.FAILED
                result.errors.append(f"Deduplication failed with code {proc.returncode}")

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.errors.append(str(e))

        result.end_time = time.time()
        return result

    def _run_export(self, task_id: str) -> TaskResult:
        """Run Blender export."""
        result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )

        mesh_dir = Path(self.config.work_dir) / task_id / "deduplicated"
        if not mesh_dir.exists():
            mesh_dir = Path(self.config.work_dir) / task_id / "extracted"

        output_base = Path(self.config.output_dir) / task_id
        output_base.mkdir(parents=True, exist_ok=True)

        blender_script = Path(__file__).parent / "core_blender.py"

        cmd = [
            self.config.blender_path, "-b", "--python", str(blender_script),
            "--",
            "--meshes", str(mesh_dir),
            "--scale", str(self.config.scale_factor)
        ]

        if self.config.export_gltf:
            cmd.extend(["--gltf", str(output_base / f"{task_id}.glb")])
        if self.config.export_usd:
            cmd.extend(["--usd", str(output_base / f"{task_id}.usdc")])
        if self.config.export_blend:
            cmd.extend(["--blend", str(output_base / f"{task_id}.blend")])
        if self.config.use_unlit:
            cmd.append("--unlit")

        try:
            log_file = Path(self.config.log_dir) / f"{task_id}_export.log"

            with open(log_file, 'w') as log:
                proc = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800  # 30 minutes for large scenes
                )

            if proc.returncode == 0:
                result.status = TaskStatus.COMPLETED

                # Collect output files
                for pattern in ["*.glb", "*.gltf", "*.usdc", "*.usd", "*.blend"]:
                    result.output_files.extend([str(f) for f in output_base.glob(pattern)])
            else:
                result.status = TaskStatus.FAILED
                result.errors.append(f"Export failed with code {proc.returncode}")

        except subprocess.TimeoutExpired:
            result.status = TaskStatus.FAILED
            result.errors.append("Export timed out after 30 minutes")
        except Exception as e:
            result.status = TaskStatus.FAILED
            result.errors.append(str(e))

        result.end_time = time.time()
        return result

    def _process_single_task(self,
                            task_id: str,
                            checkpoint_callback: Optional[Callable] = None,
                            auto_approve: bool = False) -> TaskResult:
        """Process a single capture through the full pipeline."""
        task = self.tasks[task_id]
        capture_path = task['capture_path']

        print(f"\n{'='*60}")
        print(f"Processing: {task_id}")
        print(f"Capture: {capture_path}")
        print(f"{'='*60}")

        final_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.RUNNING,
            start_time=time.time()
        )

        try:
            # Step 1: Extraction
            print("\n[1/3] Extracting meshes from RenderDoc capture...")
            extract_result = self._run_extraction(task_id, capture_path)

            if extract_result.status == TaskStatus.FAILED:
                final_result.status = TaskStatus.FAILED
                final_result.errors = extract_result.errors
                return final_result

            print(f"  Extracted {extract_result.mesh_count} meshes")

            # Checkpoint after extraction
            if self.config.checkpoint_after_extract:
                checkpoint = self._create_checkpoint(
                    CheckpointType.EXTRACTION_COMPLETE,
                    task_id,
                    {
                        'mesh_count': extract_result.mesh_count,
                        'texture_count': extract_result.texture_count,
                        'duration': extract_result.duration
                    },
                    preview_files=extract_result.output_files[:10]
                )

                if not self._wait_for_approval(checkpoint, checkpoint_callback, auto_approve):
                    final_result.status = TaskStatus.REJECTED
                    final_result.errors.append("Rejected at extraction checkpoint")
                    return final_result

            # Step 2: Deduplication
            print("\n[2/3] Deduplicating meshes...")
            dedup_result = self._run_deduplication(task_id)

            if dedup_result.status == TaskStatus.FAILED:
                # Continue with non-deduplicated meshes
                print("  Deduplication failed, using raw meshes")
                final_result.warnings.append("Deduplication failed")
            else:
                removed = dedup_result.metrics.get('duplicates_removed', 0)
                print(f"  Removed {removed} duplicates, {dedup_result.mesh_count} unique meshes")

            # Checkpoint after dedup
            if self.config.checkpoint_after_dedup and dedup_result.status == TaskStatus.COMPLETED:
                checkpoint = self._create_checkpoint(
                    CheckpointType.DEDUPLICATION_COMPLETE,
                    task_id,
                    dedup_result.metrics,
                    preview_files=dedup_result.output_files[:10]
                )

                if not self._wait_for_approval(checkpoint, checkpoint_callback, auto_approve):
                    final_result.status = TaskStatus.REJECTED
                    return final_result

            # Checkpoint before export
            if self.config.checkpoint_before_export:
                checkpoint = self._create_checkpoint(
                    CheckpointType.PRE_EXPORT,
                    task_id,
                    {
                        'export_gltf': self.config.export_gltf,
                        'export_usd': self.config.export_usd,
                        'export_blend': self.config.export_blend,
                        'mesh_count': dedup_result.mesh_count or extract_result.mesh_count
                    }
                )

                if not self._wait_for_approval(checkpoint, checkpoint_callback, auto_approve):
                    final_result.status = TaskStatus.REJECTED
                    return final_result

            # Step 3: Export
            print("\n[3/3] Exporting to target formats...")
            export_result = self._run_export(task_id)

            if export_result.status == TaskStatus.FAILED:
                final_result.status = TaskStatus.FAILED
                final_result.errors = export_result.errors
                return final_result

            print(f"  Exported {len(export_result.output_files)} files")

            # Checkpoint after export
            if self.config.checkpoint_after_export:
                checkpoint = self._create_checkpoint(
                    CheckpointType.EXPORT_COMPLETE,
                    task_id,
                    {'output_files': export_result.output_files},
                    preview_files=export_result.output_files
                )

                if not self._wait_for_approval(checkpoint, checkpoint_callback, auto_approve):
                    final_result.status = TaskStatus.REJECTED
                    return final_result

            # Success
            final_result.status = TaskStatus.COMPLETED
            final_result.output_files = export_result.output_files
            final_result.mesh_count = dedup_result.mesh_count or extract_result.mesh_count
            final_result.texture_count = extract_result.texture_count
            final_result.metrics = {
                'extraction': extract_result.metrics,
                'deduplication': dedup_result.metrics if dedup_result.status == TaskStatus.COMPLETED else {},
                'total_duration': time.time() - final_result.start_time
            }

        except Exception as e:
            final_result.status = TaskStatus.FAILED
            final_result.errors.append(f"Unexpected error: {e}")
            final_result.errors.append(traceback.format_exc())

        final_result.end_time = time.time()
        return final_result

    def run(self,
            checkpoint_callback: Optional[Callable[[Checkpoint], bool]] = None,
            auto_approve: bool = False,
            resume: bool = True) -> Dict[str, TaskResult]:
        """
        Run the batch processing.

        Args:
            checkpoint_callback: Function called at each checkpoint, returns True to approve
            auto_approve: Skip all checkpoints (for fully automated runs)
            resume: Resume from saved state if available

        Returns:
            Dict mapping task_id -> TaskResult
        """
        if resume and self.config.resume_from_checkpoint:
            self._load_state()

        pending_tasks = [
            tid for tid, task in self.tasks.items()
            if task['status'] == TaskStatus.PENDING.value
        ]

        print(f"\nBatch Processing: {len(pending_tasks)} tasks")
        print(f"Checkpoints: {'Auto-approve' if auto_approve else 'Manual review'}")
        print()

        start_time = time.time()

        if self.config.max_parallel > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_parallel) as executor:
                futures = {
                    executor.submit(
                        self._process_single_task,
                        tid,
                        checkpoint_callback,
                        auto_approve
                    ): tid
                    for tid in pending_tasks
                }

                for future in as_completed(futures):
                    task_id = futures[future]
                    try:
                        result = future.result()
                        self.results[task_id] = result
                        self.tasks[task_id]['status'] = result.status.value
                    except Exception as e:
                        print(f"Task {task_id} failed: {e}")
        else:
            # Sequential processing
            for task_id in pending_tasks:
                result = self._process_single_task(task_id, checkpoint_callback, auto_approve)
                self.results[task_id] = result
                self.tasks[task_id]['status'] = result.status.value
                self._save_state()

        # Final summary
        total_time = time.time() - start_time
        self._print_summary(total_time)

        # Save final state
        self._save_state()

        # Notifications
        self._send_notifications()

        return self.results

    def _print_summary(self, total_time: float):
        """Print batch processing summary."""
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)

        completed = sum(1 for r in self.results.values() if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED)
        rejected = sum(1 for r in self.results.values() if r.status == TaskStatus.REJECTED)

        print(f"Total tasks: {len(self.tasks)}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Rejected: {rejected}")
        print(f"Total time: {total_time:.1f} seconds")
        print()

        total_meshes = sum(r.mesh_count for r in self.results.values())
        total_textures = sum(r.texture_count for r in self.results.values())
        total_outputs = sum(len(r.output_files) for r in self.results.values())

        print(f"Total meshes processed: {total_meshes}")
        print(f"Total textures: {total_textures}")
        print(f"Output files: {total_outputs}")

        if failed > 0:
            print("\nFailed tasks:")
            for tid, result in self.results.items():
                if result.status == TaskStatus.FAILED:
                    print(f"  - {tid}: {result.errors[0] if result.errors else 'Unknown error'}")

    def _send_notifications(self):
        """Send completion notifications if configured."""
        if self.config.webhook_url:
            try:
                import urllib.request
                data = json.dumps({
                    'event': 'batch_complete',
                    'tasks': len(self.tasks),
                    'results': {k: v.to_dict() for k, v in self.results.items()}
                }).encode()

                req = urllib.request.Request(
                    self.config.webhook_url,
                    data=data,
                    headers={'Content-Type': 'application/json'}
                )
                urllib.request.urlopen(req, timeout=10)
            except Exception as e:
                print(f"Failed to send webhook: {e}")

    def generate_report(self, output_path: str):
        """Generate detailed HTML report."""
        html = ['<!DOCTYPE html><html><head>']
        html.append('<title>Batch Processing Report</title>')
        html.append('<style>')
        html.append('body { font-family: Arial, sans-serif; margin: 20px; }')
        html.append('table { border-collapse: collapse; width: 100%; }')
        html.append('th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }')
        html.append('th { background-color: #4CAF50; color: white; }')
        html.append('.completed { background-color: #dff0d8; }')
        html.append('.failed { background-color: #f2dede; }')
        html.append('.rejected { background-color: #fcf8e3; }')
        html.append('</style></head><body>')

        html.append('<h1>Batch Processing Report</h1>')
        html.append(f'<p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')

        html.append('<h2>Summary</h2>')
        html.append('<table><tr>')
        html.append('<th>Total Tasks</th><th>Completed</th><th>Failed</th><th>Rejected</th>')
        html.append('</tr><tr>')

        completed = sum(1 for r in self.results.values() if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in self.results.values() if r.status == TaskStatus.FAILED)
        rejected = sum(1 for r in self.results.values() if r.status == TaskStatus.REJECTED)

        html.append(f'<td>{len(self.tasks)}</td>')
        html.append(f'<td>{completed}</td>')
        html.append(f'<td>{failed}</td>')
        html.append(f'<td>{rejected}</td>')
        html.append('</tr></table>')

        html.append('<h2>Task Details</h2>')
        html.append('<table><tr>')
        html.append('<th>Task ID</th><th>Status</th><th>Meshes</th><th>Duration</th><th>Outputs</th>')
        html.append('</tr>')

        for task_id, result in self.results.items():
            status_class = result.status.value
            html.append(f'<tr class="{status_class}">')
            html.append(f'<td>{task_id}</td>')
            html.append(f'<td>{result.status.value}</td>')
            html.append(f'<td>{result.mesh_count}</td>')
            html.append(f'<td>{result.duration:.1f}s</td>')
            html.append(f'<td>{len(result.output_files)}</td>')
            html.append('</tr>')

        html.append('</table>')
        html.append('</body></html>')

        with open(output_path, 'w') as f:
            f.write('\n'.join(html))

        print(f"Report saved: {output_path}")


# Convenience functions
def run_batch(captures: List[str],
              output_dir: str = "export",
              auto_approve: bool = False,
              config: Optional[BatchConfig] = None) -> Dict[str, TaskResult]:
    """
    Simple batch processing function.

    Args:
        captures: List of .rdc capture paths
        output_dir: Output directory
        auto_approve: Skip manual checkpoints
        config: Optional BatchConfig

    Returns:
        Dict of results
    """
    if config is None:
        config = BatchConfig(output_dir=output_dir)

    processor = BatchProcessor(config)
    processor.add_captures(captures)
    return processor.run(auto_approve=auto_approve)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch process RenderDoc captures")
    parser.add_argument('captures', nargs='+', help="RenderDoc capture files or directory")
    parser.add_argument('--output', '-o', default='export', help="Output directory")
    parser.add_argument('--config', '-c', help="Config file path")
    parser.add_argument('--auto', action='store_true', help="Auto-approve all checkpoints")
    parser.add_argument('--parallel', '-p', type=int, default=1, help="Parallel tasks")
    parser.add_argument('--report', '-r', help="Generate HTML report")

    args = parser.parse_args()

    # Load or create config
    if args.config and Path(args.config).exists():
        config = BatchConfig.load(args.config)
    else:
        config = BatchConfig(output_dir=args.output, max_parallel=args.parallel)

    processor = BatchProcessor(config)

    # Add captures
    for capture in args.captures:
        path = Path(capture)
        if path.is_dir():
            processor.add_captures_from_directory(str(path))
        elif path.exists():
            processor.add_capture(str(path))
        else:
            print(f"Warning: {capture} not found")

    # Run
    results = processor.run(auto_approve=args.auto)

    # Generate report if requested
    if args.report:
        processor.generate_report(args.report)
