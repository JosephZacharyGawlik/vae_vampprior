# 🛠 VAE Project: GCP Setup & Sync Guide

## 1. Instance Identity
- **Instance Name:** `instance-20260116-104440`
- **Recommended Spec:** `n1-standard-4` + `NVIDIA T4` (Spot VM)

---

## 2. SSH Setup (Local & VS Code)
### Step A: Authorize Local Key
1. Local terminal: `pbcopy < ~/.ssh/id_ed25519.pub`. This copies your public key.
2. GCP Console: **Compute Engine > Metadata > SSH Keys > Add Item**. Paste key.

### Step B: SSH Shortcut
Add to local `~/.ssh/config`:
```text
Host vae-box
    HostName [VM_EXTERNAL_IP]
    User josephzgawlik
    IdentityFile ~/.ssh/id_ed25519
```
> Replace [VM_EXTERNAL_IP] with the external IP of your VM.

Run this in Terminal:
```bash
# 1. Install Git
sudo apt update && sudo apt install git -y

# 2. Install uv (Fast Python manager)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# 3. Update Path (Crucial step!)
source $HOME/.local/bin/env


# Verify
uv --version
```

## 3. Clone GitHub Repo

This one that is.

## 4. Set Up Environment

```bash
cd ~/vae_vampprior

# Create venv and install all deps automatically
uv sync

# Activate the environment
source .venv/bin/activate
```

## Change Disk Space

You'll likely need more storage. If you already created an instance, use these steps:

1. Go to the Google Cloud Disks page.

2. Click on the name of your disk (it usually matches your instance name).

3. Click Edit at the top.

4. Change the Size (I used 40GB).

5. Click Save.