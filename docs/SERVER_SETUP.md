# Server & Container Setup

This project targets an NVIDIA **Blackwell** GPU (e.g. RTX PRO 6000, `sm_120`) on
**CUDA 12.8 / Ubuntu 24.04**. The reference environment is a Docker container: you bring it
up, SSH in, and run the normal [`setup.sh`](../setup.sh). For the GPU/CUDA/ABI reasoning,
see [`ADVANCED_NVBLOX.md`](./ADVANCED_NVBLOX.md).

## 1. The container (`docker-compose.yml`)

A ready-to-run compose file lives at the repo root:

```yaml
services:
  ai-workspace:
    image: nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
    container_name: rafael_workspace
    restart: unless-stopped
    ports:
      - "2222:22"
    volumes:
      - ./container_workspace:/root
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      # Silences the uv wheel naming error globally
      - UV_SKIP_WHEEL_FILENAME_CHECK=1
    command: >
      bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server git curl libgl1 libglib2.0-0 python3.12 python3.12-venv python3-pip ninja-build
      && mkdir /var/run/sshd
      && echo 'root:lab_password' | chpasswd
      && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
      && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config
      && /usr/sbin/sshd -D"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

What each part does:

| Setting | Why it matters |
| --- | --- |
| `image: nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04` | The `-devel` tag ships `nvcc` (needed to build nvblox from source). CUDA 12.8 is the first toolkit with Blackwell `sm_120` kernels. |
| `UV_SKIP_WHEEL_FILENAME_CHECK=1` | Set container-wide so `uv` accepts nvblox's non-manylinux wheel filename; without it `uv sync` fails with a wheel-name validation error. |
| `volumes: ./container_workspace:/root` | Your home — the cloned repo, the uv cache, and the `.venv` — persists on the host across restarts. |
| `ports: "2222:22"` | SSH is published on host port **2222**. |
| `deploy.resources...nvidia` | Passes all host GPUs into the container. |
| `command: ... sshd -D` | One-time install of OpenSSH + the Python 3.12 toolchain, then runs the SSH daemon in the foreground. |

> **Recommended addition:** also set `UV_PYTHON_PREFERENCE=system` under `environment:` so
> `uv` uses the apt-installed Python 3.12 instead of trying to download a managed interpreter
> (which fails on hosts without GitHub access). `setup.sh` exports this anyway, so it's
> optional but convenient for manual `uv` calls.
>
> **Security:** `root:lab_password` plus permissive SSH config is fine for a **trusted lab
> network only**. Change the password (and prefer key-based auth) before exposing this host.

Bring it up from the directory containing `docker-compose.yml`:

```bash
docker compose up -d
docker compose logs -f          # watch the one-time apt install, until you see sshd start
```

## 2. SSH into the container

```bash
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no root@localhost -p 2222
# Password: lab_password
```

The two `-o` flags skip host-key checking. That's deliberate: the container generates a new
SSH host key every time it's recreated, so without them SSH would refuse to reconnect with a
`REMOTE HOST IDENTIFICATION HAS CHANGED` warning. Replace `localhost` with the server's
hostname or IP when connecting from another machine.

## 3. Install the project

Inside the container:

```bash
git clone --recurse-submodules https://github.com/zRafaF/PRISM-VGGT
cd PRISM-VGGT
chmod +x setup.sh
./setup.sh
```

`setup.sh` installs `uv`, initializes the PanoVGGT submodule, syncs the CUDA 12.8 / torch 2.8
environment, and downloads the model weights. It installs `nvblox_torch` in one of three modes:

| Mode | How to select | What it does |
| --- | --- | --- |
| `prebuilt` *(default)* | `./setup.sh` or `./setup.sh prebuilt` | Installs the wheel pinned in `pyproject.toml`. Fastest, but **segfaults on Blackwell + cu128** (C++ ABI mismatch). |
| `source` | `./setup.sh source` | Builds nvblox from source via `scripts/build_nvblox.sh`, matched to your GPU arch, CUDA version, and torch ABI. **Use this on the RTX PRO 6000.** |
| `url` | `./setup.sh url` | Installs a specific pre-built wheel you provide (prompts for the URL, or set `NVBLOX_WHEEL_URL`). |

Selection priority is: **first CLI argument → `NVBLOX_MODE` env var → interactive menu (60s
timeout) → `prebuilt`**. So it also runs fully unattended:

```bash
NVBLOX_MODE=source ./setup.sh                                  # build from source, no prompts
NVBLOX_MODE=url NVBLOX_WHEEL_URL="https://.../my.whl" ./setup.sh
```

> On the Blackwell server, choose **`source`**. After it builds, pin the local build so a
> later `uv sync` won't reinstall the broken prebuilt wheel over it:
>
> ```toml
> [tool.uv.sources]
> nvblox-torch = { path = "nvblox/nvblox_torch", editable = true }
> ```

## 4. Launch the app

```bash
uv sync --extra apps
uv run apps/gradio_ui.py          # binds 0.0.0.0:7860
```

The Gradio UI listens on `0.0.0.0:7860`. The reference compose only publishes the SSH port,
so reach the UI either by adding `- "7860:7860"` to `ports:` (then `docker compose up -d`
again), or by forwarding it over SSH:

```bash
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -L 7860:localhost:7860 root@localhost -p 2222
# then open http://localhost:7860 in your browser
```
