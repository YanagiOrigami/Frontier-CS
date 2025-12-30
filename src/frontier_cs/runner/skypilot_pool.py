"""
SkyPilot Pool Runner - Uses managed jobs pool for faster batch evaluation.

Instead of launching a new cluster for each evaluation, this runner:
1. Creates a pool of workers with identical environments
2. Submits jobs to the pool for execution
3. Workers are reused across jobs, avoiding cold start overhead

This can reduce evaluation time by 10-20x for large batch runs.
"""

import hashlib
import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import EvaluationResult, EvaluationStatus, RunnerBase
from .config import load_runtime_config

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for a SkyPilot pool."""
    name: str
    workers: int = 4
    accelerators: str = "A100:1"
    cpus: str = "8+"
    memory: str = "32+"
    disk_size: int = 100
    cloud: Optional[str] = None
    region: Optional[str] = None
    idle_minutes: int = 30  # Auto-stop after idle


@dataclass
class JobResult:
    """Result of a managed job."""
    job_id: int
    pair_id: str
    status: str  # PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED
    score: Optional[float] = None
    score_unbounded: Optional[float] = None
    message: Optional[str] = None
    duration_seconds: Optional[float] = None


class SkyPilotPoolRunner(RunnerBase):
    """
    Runner that uses SkyPilot managed jobs pool for batch evaluation.

    This runner is designed for batch evaluation of many solutions,
    where workers can be reused across jobs for faster execution.
    """

    DEFAULT_POOL_NAME = "frontier-eval-pool"
    DEFAULT_WORKERS = 4
    DEFAULT_ACCELERATORS = "A100:1"
    DEFAULT_CPUS = "8+"
    DEFAULT_MEMORY = "32+"
    DEFAULT_DISK_SIZE = 100
    DEFAULT_TIMEOUT = 1800  # 30 minutes per job

    def __init__(
        self,
        research_dir: Path,
        pool_config: Optional[PoolConfig] = None,
        timeout: Optional[int] = None,
        bucket_url: Optional[str] = None,
    ):
        """
        Initialize the pool runner.

        Args:
            research_dir: Path to research directory containing problems/
            pool_config: Pool configuration (or use defaults)
            timeout: Default timeout per job in seconds
            bucket_url: Optional GCS/S3 bucket URL for result storage
        """
        self.research_dir = Path(research_dir)
        self.pool_config = pool_config or PoolConfig(name=self.DEFAULT_POOL_NAME)
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self.bucket_url = bucket_url

        # Track submitted jobs
        self._pending_jobs: Dict[int, str] = {}  # job_id -> pair_id
        self._results: Dict[str, JobResult] = {}  # pair_id -> result

    def get_problem_path(self, problem_id: str) -> Path:
        """Get the path to a problem directory."""
        return self.research_dir / "problems" / problem_id

    def create_pool(self) -> bool:
        """
        Create or update the worker pool.

        Returns:
            True if pool was created/updated successfully
        """
        import sky

        logger.info(f"Creating pool '{self.pool_config.name}' with {self.pool_config.workers} workers")

        # Build pool YAML
        pool_yaml = self._generate_pool_yaml()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(pool_yaml)
            pool_file = f.name

        try:
            # Use CLI to create pool (SDK may not have pool apply yet)
            result = subprocess.run(
                ["sky", "jobs", "pool", "apply", pool_file, "-y"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout for pool creation
            )

            if result.returncode != 0:
                logger.error(f"Failed to create pool: {result.stderr}")
                return False

            logger.info(f"Pool '{self.pool_config.name}' created successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Pool creation timed out")
            return False
        except Exception as e:
            logger.error(f"Error creating pool: {e}")
            return False
        finally:
            Path(pool_file).unlink(missing_ok=True)

    def _generate_pool_yaml(self) -> str:
        """Generate YAML configuration for the pool."""
        cfg = self.pool_config

        yaml_content = f"""
# Auto-generated pool configuration for Frontier evaluation
pool:
  workers: {cfg.workers}

resources:
  cpus: "{cfg.cpus}"
  memory: "{cfg.memory}"
  accelerators: "{cfg.accelerators}"
  disk_size: {cfg.disk_size}
"""

        if cfg.cloud:
            yaml_content += f"  cloud: {cfg.cloud}\n"
        if cfg.region:
            yaml_content += f"  region: {cfg.region}\n"

        yaml_content += f"""
setup: |
  # Setup Docker for evaluation
  if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
  fi

  # Ensure nvidia-container-toolkit is available
  if command -v nvidia-smi &> /dev/null; then
    if ! command -v nvidia-container-cli &> /dev/null; then
      distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
      curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
      curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
      sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
      sudo systemctl restart docker || true
    fi
  fi

  echo "Pool worker setup complete"
"""
        return yaml_content

    def submit_job(
        self,
        problem_id: str,
        solution_path: Path,
        *,
        solution_id: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Optional[int]:
        """
        Submit a job to the pool for evaluation.

        Args:
            problem_id: Problem ID
            solution_path: Path to solution file
            solution_id: Optional solution identifier
            timeout: Optional timeout override

        Returns:
            Job ID if submitted successfully, None otherwise
        """
        problem_path = self.get_problem_path(problem_id)
        if not problem_path.exists():
            logger.error(f"Problem not found: {problem_path}")
            return None

        if not solution_path.exists():
            logger.error(f"Solution not found: {solution_path}")
            return None

        # Load runtime config
        runtime_config = load_runtime_config(problem_path)
        docker_config = runtime_config.docker

        # Generate job YAML
        pair_id = f"{solution_id or solution_path.name}:{problem_id}"
        job_name = self._generate_job_name(pair_id)
        job_yaml = self._generate_job_yaml(
            job_name=job_name,
            problem_id=problem_id,
            problem_path=problem_path,
            solution_path=solution_path,
            docker_image=docker_config.image,
            docker_gpu=docker_config.gpu,
            timeout=timeout or self.timeout,
            pair_id=pair_id,
        )

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False
        ) as f:
            f.write(job_yaml)
            job_file = f.name

        try:
            # Submit job to pool via CLI
            result = subprocess.run(
                [
                    "sky", "jobs", "launch",
                    "-p", self.pool_config.name,
                    "-y", "--detach",
                    job_file,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                logger.error(f"Failed to submit job: {result.stderr}")
                return None

            # Parse job ID from output
            job_id = self._parse_job_id(result.stdout)
            if job_id:
                self._pending_jobs[job_id] = pair_id
                logger.info(f"Submitted job {job_id} for {pair_id}")
                return job_id
            else:
                logger.error("Could not parse job ID from output")
                return None

        except Exception as e:
            logger.error(f"Error submitting job: {e}")
            return None
        finally:
            Path(job_file).unlink(missing_ok=True)

    def _generate_job_name(self, pair_id: str) -> str:
        """Generate a unique job name from pair ID."""
        digest = hashlib.md5(pair_id.encode()).hexdigest()[:8]
        # Sanitize and truncate
        name = pair_id.replace(":", "-").replace("/", "-").replace(".", "-")
        name = "".join(c if c.isalnum() or c == "-" else "-" for c in name)
        return f"eval-{name[:40]}-{digest}"

    def _generate_job_yaml(
        self,
        job_name: str,
        problem_id: str,
        problem_path: Path,
        solution_path: Path,
        docker_image: str,
        docker_gpu: bool,
        timeout: int,
        pair_id: str,
    ) -> str:
        """Generate YAML configuration for a job."""

        gpu_flag = "--gpus all" if docker_gpu else ""

        yaml_content = f"""
name: {job_name}

file_mounts:
  ~/problem:
    source: {problem_path}
    mode: COPY
  ~/solution.py:
    source: {solution_path}
    mode: COPY

run: |
  set -e
  mkdir -p ~/results

  # Copy solution to problem directory
  cp ~/solution.py ~/problem/solution.py

  # Run evaluation in Docker
  cd ~/problem

  if [ -f evaluate.sh ]; then
    docker run --rm {gpu_flag} \\
      -v $(pwd):/work \\
      -w /work \\
      {docker_image} \\
      bash -c "chmod +x evaluate.sh && ./evaluate.sh" \\
      2>&1 | tee ~/results/output.txt
  else
    docker run --rm {gpu_flag} \\
      -v $(pwd):/work \\
      -w /work \\
      {docker_image} \\
      python evaluator.py --solution solution.py \\
      2>&1 | tee ~/results/output.txt
  fi

  # Extract score (last line with numbers)
  grep -E "^-?[0-9]+\\.?[0-9]*(\\s+-?[0-9]+\\.?[0-9]*)?$" ~/results/output.txt | tail -1 > ~/results/score.txt || true

  echo "Evaluation complete for {pair_id}"
"""
        return yaml_content

    def _parse_job_id(self, output: str) -> Optional[int]:
        """Parse job ID from sky jobs launch output."""
        import re
        # Look for patterns like "Job ID: 123" or "Managed Job ID: 123"
        match = re.search(r"(?:Managed )?Job ID:\s*(\d+)", output)
        if match:
            return int(match.group(1))
        # Also try "Jobs submitted with IDs: 1,2,3"
        match = re.search(r"Jobs submitted with IDs?:\s*([\d,]+)", output)
        if match:
            ids = match.group(1).split(",")
            return int(ids[0].strip())
        return None

    def get_job_status(self, job_id: int) -> Optional[JobResult]:
        """
        Get the status of a submitted job.

        Args:
            job_id: Job ID returned from submit_job

        Returns:
            JobResult with current status
        """
        try:
            result = subprocess.run(
                ["sky", "jobs", "queue", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return None

            import json
            jobs = json.loads(result.stdout)

            for job in jobs:
                if job.get("job_id") == job_id:
                    pair_id = self._pending_jobs.get(job_id, f"unknown:{job_id}")
                    status = job.get("status", "UNKNOWN")

                    return JobResult(
                        job_id=job_id,
                        pair_id=pair_id,
                        status=status,
                    )

            return None

        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None

    def wait_for_jobs(
        self,
        job_ids: List[int],
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> Dict[int, JobResult]:
        """
        Wait for multiple jobs to complete.

        Args:
            job_ids: List of job IDs to wait for
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait (None for no limit)

        Returns:
            Dict mapping job_id to JobResult
        """
        results: Dict[int, JobResult] = {}
        pending = set(job_ids)
        start_time = time.time()

        while pending:
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for jobs: {pending}")
                break

            for job_id in list(pending):
                job_result = self.get_job_status(job_id)
                if job_result:
                    if job_result.status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                        # Job completed, fetch results
                        self._fetch_job_results(job_result)
                        results[job_id] = job_result
                        pending.discard(job_id)
                        logger.info(f"Job {job_id} completed: {job_result.status}")

            if pending:
                time.sleep(poll_interval)

        return results

    def _fetch_job_results(self, job_result: JobResult) -> None:
        """Fetch score and logs from completed job."""
        try:
            # Get job logs
            result = subprocess.run(
                ["sky", "jobs", "logs", str(job_result.job_id)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                logs = result.stdout
                # Parse score from logs
                score, score_unbounded = self._parse_score_from_logs(logs)
                job_result.score = score
                job_result.score_unbounded = score_unbounded

        except Exception as e:
            logger.error(f"Error fetching job results: {e}")
            job_result.message = str(e)

    def _parse_score_from_logs(self, logs: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse score from job logs."""
        import re

        # Look for score line (last line with numbers)
        lines = logs.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            match = re.match(r"^(-?\d+\.?\d*)\s*(-?\d+\.?\d*)?$", line)
            if match:
                score = float(match.group(1))
                score_unbounded = float(match.group(2)) if match.group(2) else score
                return score, score_unbounded

        return None, None

    def stop_pool(self) -> bool:
        """
        Stop the worker pool and terminate all workers.

        Returns:
            True if pool was stopped successfully
        """
        try:
            result = subprocess.run(
                ["sky", "jobs", "pool", "delete", self.pool_config.name, "-y"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                logger.info(f"Pool '{self.pool_config.name}' stopped")
                return True
            else:
                logger.error(f"Failed to stop pool: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error stopping pool: {e}")
            return False

    def evaluate(
        self,
        problem_id: str,
        solution_code: str,
        *,
        timeout: Optional[int] = None,
    ) -> EvaluationResult:
        """
        Evaluate a solution (single job, synchronous).

        For batch evaluation, use submit_job() and wait_for_jobs() instead.
        """
        # Write solution to temp file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(solution_code)
            solution_path = Path(f.name)

        try:
            return self.evaluate_file(
                problem_id,
                solution_path,
                timeout=timeout,
            )
        finally:
            solution_path.unlink(missing_ok=True)

    def evaluate_file(
        self,
        problem_id: str,
        solution_path: Path,
        *,
        timeout: Optional[int] = None,
        solution_id: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Evaluate a solution file (single job, synchronous).

        For batch evaluation, use submit_job() and wait_for_jobs() instead.
        """
        start_time = time.time()

        # Submit job
        job_id = self.submit_job(
            problem_id,
            solution_path,
            solution_id=solution_id,
            timeout=timeout,
        )

        if job_id is None:
            return EvaluationResult(
                problem_id=problem_id,
                status=EvaluationStatus.ERROR,
                message="Failed to submit job to pool",
            )

        # Wait for completion
        results = self.wait_for_jobs(
            [job_id],
            timeout=timeout or self.timeout,
        )

        duration = time.time() - start_time

        if job_id not in results:
            return EvaluationResult(
                problem_id=problem_id,
                status=EvaluationStatus.TIMEOUT,
                message="Job timed out",
                duration_seconds=duration,
            )

        job_result = results[job_id]

        if job_result.status == "SUCCEEDED":
            return EvaluationResult(
                problem_id=problem_id,
                status=EvaluationStatus.SUCCESS,
                score=job_result.score,
                score_unbounded=job_result.score_unbounded,
                duration_seconds=duration,
            )
        else:
            return EvaluationResult(
                problem_id=problem_id,
                status=EvaluationStatus.ERROR,
                message=job_result.message or f"Job failed with status: {job_result.status}",
                duration_seconds=duration,
            )
