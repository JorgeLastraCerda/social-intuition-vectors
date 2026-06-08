# SCCKN Cluster — Windows Setup Guide

This file documents how to connect to the SCCKN cluster from a Windows PC and how to
run jobs. It mirrors `archive/target_self_affect_leakage/scckn/SCCKN_SETUP.md` (the
original Mac setup), adapted for Windows 11 with the built-in OpenSSH client.

---

## Account Details

| Field           | Value                                   |
|-----------------|-----------------------------------------|
| Username        | emrecan.ulu                             |
| Frontend 1      | scc.uni-konstanz.de                     |
| Frontend 2      | scc2.uni-konstanz.de                    |
| Home directory  | /home/scc/emrecan.ulu  (100 GB quota)   |
| Work storage    | /work/emrecan.ulu  (large data, no backup guarantee) |
| Scratch         | /localscratch  (node-local, no backup)  |
| JupyterHub      | https://scc2.uni-konstanz.de            |
| Sysadmin        | Stefan.Gerlach@uni-konstanz.de          |

---

## 1. SSH Key

Windows 11 ships with OpenSSH. No install needed — `ssh` and `ssh-keygen` are at
`C:\Windows\System32\OpenSSH\`.

An `ed25519` key pair already exists at:

```
C:\Users\emrec\.ssh\id_ed25519        (private — never share)
C:\Users\emrec\.ssh\id_ed25519.pub    (public — copy to cluster)
```

If you ever need to regenerate a key:

```powershell
ssh-keygen -t ed25519 -C "scckn"
```

---

## 2. Copy Public Key to Cluster (one-time setup)

Run the following in a terminal (PowerShell or CMD). You will be prompted for your
cluster password once:

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh emrecan.ulu@scc.uni-konstanz.de "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

After this, password prompts stop and key-based login is used automatically.

---

## 3. SSH Config (alias)

`C:\Users\emrec\.ssh\config` already contains:

```
Host scckn
    HostName scc.uni-konstanz.de
    User emrecan.ulu
    IdentityFile ~/.ssh/id_ed25519
```

This lets you connect with just:

```powershell
ssh scckn
```

Test it (should print hostname and your username without a password prompt):

```powershell
ssh scckn "hostname; whoami"
```

---

## 4. File Transfer

Code is synced via git (see Section 5). For large data files (activations, results):

```powershell
# Windows → Cluster
scp path\to\file.npy scckn:/work/emrecan.ulu/normalcy-axis/

# Cluster → Windows
scp scckn:/work/emrecan.ulu/normalcy-axis/results/file.json .
```

For folder transfers add `-r`. For large batches, `rsync` is faster but requires
the Windows Subsystem for Linux (WSL) or Git Bash — see archive STORAGE.md.

---

## 5. Code Sync (git)

The repo is on GitHub (`github.com/JorgeLastraCerda/normalcy-axis`). The cluster copy
uses an SSH remote so push/pull works without PAT tokens. This was set up by adding a
GitHub deploy key (`~/.ssh/id_ed25519_github`) and running:

```bash
git remote set-url origin git@github.com:JorgeLastraCerda/normalcy-axis.git
```

On the cluster the repo lives at `/work/emrecan.ulu/normalcy-axis`. Normal workflow:

```bash
# Push from Windows, then on cluster:
cd /work/emrecan.ulu/normalcy-axis
git pull

# Or push directly from cluster (SSH key already in GitHub):
git add . && git commit -m "..." && git push
```

`~/.ssh/known_hosts` already contains `github.com`; no further setup needed.

---

## 6. Python Environment

The cluster's base conda environment `python-3.13` has numpy/scipy/sklearn but not our
deep-learning packages. We maintain two project-specific environments:

| Environment | Packages                         | Used for                    |
|-------------|----------------------------------|-----------------------------|
| `wc-tl`     | transformer-lens, sae-lens, torch | Gemma 3 + GemmaScope 2 tests |
| `wc-nn`     | nnsight, nnterp, torch           | Gemma 4 tests               |

Create them on the cluster:

```bash
module load conda

# Gemma 3 environment
conda create -n wc-tl python=3.11 -y
conda activate wc-tl
pip install torch transformer-lens sae-lens transformers accelerate tqdm numpy pandas scikit-learn scipy matplotlib seaborn pyyaml

# Gemma 4 environment
conda create -n wc-nn python=3.11 -y
conda activate wc-nn
pip install torch nnsight nnterp transformers accelerate tqdm numpy pandas scikit-learn scipy matplotlib seaborn pyyaml
```

Set HuggingFace cache to `/work` (not home, which has a 100 GB quota).
`~/.bashrc` already contains `export HF_HOME=/work/emrecan.ulu/hf_cache` — skip if
already set, check with `echo $HF_HOME`.

```bash
echo 'export HF_HOME=/work/emrecan.ulu/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

Gemma 3 models are gated — accept the license on the HuggingFace model page then log in.
Gemma 4 models (`google/gemma-4-12B-it`) are Apache 2.0 and not gated, but logging in
is still needed for any gated model in the same session:

```bash
conda activate wc-tl
huggingface-cli login       # paste your HF token when prompted
huggingface-cli whoami      # verify login
```

GPU node pinning: `-l gpu=1` alone may schedule your job on a small node (11 GB VRAM).
For 12B models (~24 GB bf16) pin to L40 nodes:

```
#$ -q gpu@scc213     # L40, 48 GB, 8 GPUs
#$ -q gpu@scc192     # L40, 48 GB, 4 GPUs
```

For 27B models (~54 GB bf16) use scc214 (RTX 6000, 96 GB):

```
#$ -q gpu@scc214
```

---

## 7. Submitting GPU Jobs (SGE)

SCCKN uses Grid Engine (`qsub`, not `sbatch`). Job scripts are in `jobs/sge/`.

```bash
qsub jobs/sge/smoke_gemma3.sh      # submit a job
qstat -u emrecan.ulu               # check your jobs
qdel <job_id>                      # cancel a job
qacct -j <job_id>                  # check resource usage after completion
```

Minimal GPU job script template (exact GPU resource flags are `# ADJUST` — confirm
with `qconf` or ask Stefan):

```bash
#!/bin/bash
#$ -N job_name
#$ -q gpu                          # GPU queue
#$ -l h_rt=02:00:00                # max runtime
#$ -l h_vmem=32G                   # RAM per slot
#$ -pe smp 4                       # CPU cores
#$ -l gpu=1                        # # ADJUST: exact GPU resource flag
#$ -o logs/job_name.out
#$ -e logs/job_name.err
#$ -cwd
#$ -m ea
#$ -M emrecan.ulu@uni-konstanz.de

module load conda
conda activate wc-tl               # or wc-nn for Gemma 4
export HF_HOME=/work/emrecan.ulu/hf_cache

cd /work/emrecan.ulu/normalcy-axis
python smoke_tests/gemma3_transformerlens/smoke_test_probe.py
```

Verify GPU is available in a quick test job:

```python
import torch; print(torch.cuda.get_device_name(0))
```

---

## 8. Interactive Sessions (tmux)

Always use tmux so work survives SSH disconnects:

```bash
tmux new -s work          # start a session
# Ctrl+B then D           # detach (keeps running in background)
tmux attach -t work       # reattach later
tmux ls                   # list sessions
```

---

## 9. Off-Campus Access

The cluster is directly reachable on `scc.uni-konstanz.de:22` from the university
network. From off-campus you may need the **Uni Konstanz VPN**. If `ssh scckn` times
out at home, connect via VPN first.

---

## Quick Reference

```powershell
# Connect
ssh scckn

# Run a job
# (SSH into cluster first, then:)
qsub jobs/sge/smoke_gemma3.sh
qstat -u emrecan.ulu
qdel <job_id>
```
