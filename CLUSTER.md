to ssh: ssh -i ~/.ssh/<your_key> <cluster_username>@<head-node-ip>   (get the current username and IP from Chuning)

# Cluster quickstart

Welcome to the `ml-dev` cluster! This guide covers the essentials: connecting
to the cluster, choosing a GPU, working with storage, and running interactive or
batch jobs.

## Connect to the cluster

```bash
ssh -i ~/.ssh/your_key <username>@<head-node-ip>
```

Ask the cluster owner for the current head-node IP address. The address may
change if the cluster is rebuilt.

## How the cluster is organized

When you connect, you land on the **head node**. It is an `m7i.large` with **2
vCPUs, 8 GB of RAM, and no GPU**. Use it for lightweight tasks such as editing
code, moving data, and submitting jobs. Run training and other resource-heavy
work on a compute node.

**Compute nodes** are launched on demand by Slurm and shut down after **5 minutes
of idle time**. If no compute nodes are running, your first job may take about
2–5 minutes to start while a node boots. This delay is expected.

## Storage

| Path | Best for | Capacity | Survives cluster deletion? |
|------|----------|----------|----------------------------|
| `/home/<username>` | Code and small files; shared across all nodes (EFS) | Effectively unlimited | No |
| `/fsx` | Active datasets and checkpoints; shared across all nodes (Lustre) | 1.2 TB total | No |
| S3 buckets | Durable datasets, checkpoints, and logs | Unlimited | Yes |

> **Important:** Both `/home` and `/fsx` are working storage and are erased if
> the cluster is deleted. Copy anything you want to keep to S3.

Use `/home` for code and small files. Use `/fsx` for datasets and checkpoints
that need fast, parallel access. Because `/fsx` is shared and world-writable,
start by creating a directory for your own files:

```bash
mkdir -p /fsx/$USER
```

### S3

All nodes can read from and write to the cluster's three S3 buckets. You do not
need to provide credentials or run `aws configure`; each node already has the
required IAM role.

```bash
aws s3 ls   s3://ml-research-<ACCOUNT_ID>-datasets/
aws s3 cp   s3://ml-research-<ACCOUNT_ID>-datasets/foo.tar /fsx/$USER/
aws s3 sync /fsx/$USER/checkpoints s3://ml-research-<ACCOUNT_ID>-checkpoints/$USER/
```

| Bucket | Intended use |
|--------|--------------|
| `ml-research-<ACCOUNT_ID>-datasets` | Datasets |
| `ml-research-<ACCOUNT_ID>-checkpoints` | Model checkpoints |
| `ml-research-<ACCOUNT_ID>-logs` | Logs |

> **Take care with shared data:** These buckets are not versioned, and everyone
> can write to them. Overwritten files cannot be recovered. Keep your files
> under a `$USER` prefix, and never use `sync --delete` on a shared prefix.

## Choose a compute partition

Partitions are named after their EC2 instance families. A plain partition name,
such as `g6`, uses **on-demand** instances: they are not interrupted, but they
cost more. A name ending in `-spot`, such as `g6-spot`, uses **Spot** instances:
they are much less expensive, but AWS may reclaim them while a job is running.

| Partition | GPUs per node | Nodes | vCPUs / RAM per node |
|-----------|---------------|-------|----------------------|
| `g5` / `g5-spot` | 4 × A10G (24 GB) | 1 / 4 | 48 / 192 GB |
| `g6` / `g6-spot` | 4 × L4 (24 GB) | 1 / 2 | 48 / 192 GB |
| `g6e` / `g6e-spot` | 4 × L40S (48 GB) | 1 / 2 | 48 / 384 GB |
| `p4d` / `p4d-spot` | 8 × A100 (40 GB, NVLink) | 1 / 2 | 96 / 1152 GB |

Capability generally increases as you move down the table. For everyday work,
`g6` is a good default. Use `g6e` when you need 48 GB of GPU memory, and
reserve `p4d` for genuinely large training workloads.

> **Always specify a partition.** Slurm defaults to `g5`, which is on-demand and
> more expensive. If you omit `--partition`, your job will run on that queue.

Spot instances offer substantial savings, so prefer them when your workload can
tolerate interruption. Because a reclaimed node ends its job without warning,
save checkpoints regularly.

To see available resources and check your jobs:

```bash
sinfo                  # Show partitions and node states
squeue -u $USER        # Show your jobs
```

## Start an interactive GPU shell

Use `srun` when you want a terminal on a GPU node—for example, to explore a
dataset, debug code, or run a short experiment:

```bash
srun --partition=g6 --gres=gpu:1 \
     --cpus-per-task=12 --mem=64G --time=2:00:00 \
     --pty bash -i
```

Once the shell starts, `nvidia-smi` should show the assigned GPU. Run `exit` or
press Ctrl-D when you are finished so Slurm can release the resources.

To request all four GPUs on the node, use `--gres=gpu:4 --cpus-per-task=48
--mem=0`. Here, `--mem=0` requests all of the node's memory.

Request only the resources you expect to use. A smaller request can be easier to
schedule. Keep in mind that `--time` is a hard limit: Slurm stops the job when
the requested time expires.

## Submit a batch job

For longer or repeatable workloads, create a file named `train.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=g6-spot
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out      # <jobname>-<jobid>.out, in the submit directory

source /fsx/miniconda3/etc/profile.d/conda.sh
conda activate /fsx/$USER/envs/your_env     # your own env -- see "Set up a Python environment"

nvidia-smi
torchrun --standalone --nproc_per_node=4 train.py
```

Then submit and monitor it with:

```bash
sbatch train.sbatch          # Submit the job
squeue -u $USER              # Check its status
tail -f train-*.out          # Follow its output
scancel <jobid>              # Cancel it
```

Slurm copies the script when you submit it, so `$0` and relative paths may not
point where you expect. Use absolute paths, or add `cd "$SLURM_SUBMIT_DIR"` near
the beginning of your script.

For a single-GPU job, use `--gres=gpu:1 --cpus-per-task=12 --mem=64G` and run
`python train.py` directly. You only need `torchrun` for multi-GPU work.

All four (or eight) GPUs assigned to a job are on the same machine, so
`torchrun --standalone` is the appropriate launcher. This cluster is not set up
for multi-node jobs.

## Set up a Python environment

A shared conda installation lives at `/fsx/miniconda3`. Make `conda` available in
your shell with:

```bash
source /fsx/miniconda3/etc/profile.d/conda.sh
```

**This prints nothing, and that is normal** — it only defines the `conda`
command. Your prompt will not change and no environment is activated until you
run `conda activate`. Add that line to your `~/.bashrc` so it happens on login.

Conda envs created by other users are read-only to you, so you need to make your 
own environment under `/fsx` directory:

```bash
conda create -p /fsx/$USER/envs/your_env python=3.11
conda activate /fsx/$USER/envs/your_env
```

Keep the environment on `/fsx` rather than `/home`: every compute node mounts
`/fsx`, and package imports are much faster there.

## Before you go

- Copy important work from `/home` and `/fsx` to S3. Neither filesystem is
  backed up.
- Keep resource-heavy work off the head node. It has no GPU, only two vCPUs, and
  is shared with other users.
- Allow a few minutes for the first job to start while a compute node boots. A
  job in `CF` (configuring) or `PD` (pending) is usually waiting for this step,
  not reporting an error.
- Checkpoint Spot jobs regularly because they can be interrupted without
  warning.
- Give jobs a little extra time, but remember that `--time` is a hard limit.
- If SSH begins timing out unexpectedly, your public IP address may have
  changed. Ask the cluster owner to add the new address to the allowlist.
- If `conda` fails with a baffling `PermissionError` on some unrelated path,
  check where you are — home directories are private, so running it while your
  working directory is another user's home breaks conda's path resolution.
  `cd ~` and retry.
