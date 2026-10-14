---
name: mount-mamba-env
description: Procedure for making an existing host-level Mamba or Conda environment accessible inside the Antigravity sandbox using UI permission grants scoped strictly to the environment path.
---

# Mount Host Mamba/Conda Environment in Sandbox

Expose an existing host-level Mamba or Conda environment inside the Antigravity terminal sandbox without re-downloading packages, scoped strictly to the specific environment directory.

---

## 1. Identify the Environment Path

Find the absolute path of the target environment on your host machine:

```bash
mamba env list
# or
conda env list
```

Identify the path for your target environment:
- `<ENV_PATH>`: The full directory path to the environment (e.g., `/path/to/mamba/envs/<env_name>`).

---

## 2. Grant Read Permission in Antigravity Settings

1. Open **Settings** in Antigravity.
2. Select your project under **Projects** (or open **Permission Settings**).
3. Under **Permission Grants**, add a `read_file` entry limited strictly to the target environment directory:
   ```text
   read_file(<ENV_PATH>)
   ```
4. Save the settings. Antigravity will automatically bind-mount only this environment directory into the sandbox as read-only.

---

## 3. Verify Inside the Sandbox

Test that the environment's Python interpreter and packages are accessible within the sandbox:

```bash
# Verify binary visibility
ls -la <ENV_PATH>/bin/python

# Verify package import
<ENV_PATH>/bin/python -c "import <package_name>; print('<package_name> loaded successfully')"
```

---

## 4. Usage Inside the Sandbox

Always invoke the environment's Python binary directly (or prepend its `bin` to `PATH`) so native compiled libraries (C/C++, BLAS/MKL, CUDA) resolve correctly:

```bash
# Option 1: Direct binary execution
<ENV_PATH>/bin/python script.py

# Option 2: Prepend to PATH
export PATH="<ENV_PATH>/bin:$PATH"
python script.py
```
