# Setup

Needed on **every** machine you run this on (training or inference) --
StarCraft II itself has no cross-machine state, so this is the same on your
work PC and on a second machine (e.g. one with a real GPU).

## 1. Python 3.10

PySC2's dependency chain isn't reliably tested past 3.10/3.11. This repo
targets **3.10** specifically.

```powershell
winget install --id Python.Python.3.10 --source winget
```

## 2. Clone the repo and create a virtual environment

```powershell
git clone git@github.com:<your-username>/StarcraftCommand.git
cd StarcraftCommand
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

(SSH clone needs your SSH key present/loaded on that machine -- `ssh-add` it,
or use the HTTPS clone URL with a personal access token instead.)

## 3. Install dependencies

**PyTorch is deliberately not in `requirements.txt`** because the right
build depends on the machine:

```powershell
# CPU-only (no NVIDIA GPU, or a very old one) -- smaller download, this
# MLP-sized policy doesn't benefit much from a GPU anyway (see "GPU vs CPU"
# below), so this is the simplest default:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU present (e.g. a GTX 10-series / RTX card): install the CUDA
# build instead. Check `nvidia-smi` for your driver's max supported CUDA
# version first, then pick a matching index from
# https://pytorch.org/get-started/locally/ -- cu121 is a safe default for
# most current drivers:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then, on any machine:

```powershell
pip install -r requirements.txt
pip install -e . --no-deps
```

### GPU vs CPU

- `sb3-contrib` auto-selects CUDA if `torch.cuda.is_available()` -- no
  config needed.
- Don't expect a big speedup here: the policy/value networks are tiny
  (2x64-unit MLPs over a 236-dim vector), so the actual matrix-multiply
  work is trivial either way.
- The real bottleneck is the StarCraft II client itself stepping the game
  forward each action -- a single-threaded, real-time-simulation cost that
  a GPU doesn't touch at all.
- A GPU mainly helps once/if the network grows much larger (e.g. a future
  CNN over spatial features) or once training is parallelized across many
  concurrent `SC2Env` instances.

## 4. StarCraft II itself

- Install it, then set `SC2PATH` if it's not in the default location PySC2
  expects (`~/StarCraftII` on Linux; on Windows PySC2 auto-detects the
  standard Battle.net install path, normally
  `C:\Program Files (x86)\StarCraft II`).
- Download the official map pack and extract `Simple64.SC2Map` (and friends)
  into `<StarCraft II install>/Maps/` -- PySC2 does not fetch maps itself:

  ```powershell
  # Password-protected; downloading/extracting it means you agree to
  # Blizzard's AI and Machine Learning License.
  Invoke-WebRequest -Uri "https://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip" -OutFile Melee.zip
  Expand-Archive -Path Melee.zip -DestinationPath "C:\Program Files (x86)\StarCraft II\Maps" -Force
  # (Expand-Archive doesn't support the zip's password -- use 7-Zip or
  # `tar`/`unzip -P iagreetotheeula Melee.zip` from Git Bash instead if you
  # hit a password prompt.)
  ```

## Verifying the setup

```powershell
pytest
```

`pytest` covers everything in `src/sc2rl/env/` except the live-client parts
of `sc2_env_wrapper.py` against fakes/stubs (`tests/fakes/fake_pysc2.py`);
`tests/test_env_wrapper.py` and `tests/test_training_smoke.py` exercise the
full env -> Monitor -> DummyVecEnv -> MaskablePPO pipeline (including
action-mask auto-detection) against a stubbed `SC2Env`, and
`tests/test_training_resume.py` validates that `--resume-from` actually
continues training (weights, optimizer state, and timestep counter) rather
than restarting. See the [README](../README.md#status) for what's been
verified end-to-end against a live client.
