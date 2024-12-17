from langchain_core import tools
from pydantic import BaseModel, Field
import subprocess

class JobParameters(BaseModel):
    job_name: str = Field(default="debug", description="Name of the job")
    walltime: str = Field(default="01:00:00", description="Walltime for the job")
    nodes: int = Field(default=1, description="Number of nodes")
    ppn: int = Field(default=1, description="Processors per node")
    mem: float = Field(default=8, description="Total memory in GB")
    script_body: str = Field(default="export RASPA_DIR=/eagle/projects/HPCBot/thang/soft/RASPA2\nexport DYLD_LIBRARY_PATH=${RASPA_DIR}/lib\nexport LD_LIBRARY_PATH=${RASPA_DIR}/lib\n$RASPA_DIR/bin/simulate", description="Main body of the script to execute")
    account: str = Field(default="IQC", description="Allocation/Account name")
    queue: str = Field(default="debug", description="Queue type")
    directory: str = Field(..., description="PBS directory to run job")
class HPCJobHandle:
    def __init__(self, job_parameters: JobParameters):
        self.job_parameters = job_parameters

    def generate_script(self) -> str:
        # Generate the job script based on the job parameters
        script = f"""#!/bin/bash -l
            #PBS -l walltime={self.job_parameters.walltime}
            #PBS -l select={self.job_parameters.nodes}:npcus={self.job_parameters.ppn}:mem={self.job_parameters.mem}gb
            #PBS -l filesystems=home:eagle
            #PBS -q {self.job_parameters.queue}
            #PBS -A {self.job_parameters.account}
            #PBS -N {self.job_parameters.job_name}

            {self.job_parameters.script_body.strip()}
            """
                    return script
    def submit_job(self) -> str:

        pass
job_params = JobParameters(
job_name="RASPA",
walltime="01:00:00",
nodes=1,
ppn=32,
)
hpc_formatter = HPCJobScheduler(job_params)
job_script = hpc_formatter.generate_script()
print(job_script)
