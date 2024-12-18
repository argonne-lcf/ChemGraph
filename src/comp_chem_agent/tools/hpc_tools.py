from models.hpc import HPCJobHandle
from langchain_core import tools
import subprocess

from comp_chem_agent.models import HPCJobHandle, JobParameters

 @tool
 def submit_job(params: JobParameters, mode="local"):
    hjh = HPCJobHandle(params)

    if mode == "local":
        pbs_command = [
            "qsub",
            "-l", f"walltime={params.walltime},select=1:ncpus={params.npcus}:mem={params.mem}gb",
            "-l" filesystems=home:eagle,
            "-q", params.queue,
            "-A", params.account,
            "-N", params.job_name,
            params.script_body
        ]

        try:
            # Submit the job
            result = subprocess.run(pbs_command, capture_output=True, text=True, check=True)
            print(f"Job submitted successfully. Job ID: {result.stdout.strip()}")
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit job: {e.stderr.strip()}")
            return e.stderr.strip()

        