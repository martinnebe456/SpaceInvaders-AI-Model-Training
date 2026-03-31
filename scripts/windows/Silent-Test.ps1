Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Common.ps1")
# Silent evaluation is the fastest way to compare checkpoints numerically.
Activate-Venv
Ensure-MplConfig

# These settings can be overridden from the shell without editing the script.
$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/space_invaders_dqn.zip" }
$Episodes = if ($env:EPISODES) { $env:EPISODES } else { "10" }
$MaxSteps = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { "20000" }

python -m space_invaders_rl.evaluate `
    --model-path $ModelPath `
    --episodes $Episodes `
    --max-steps $MaxSteps `
    --render-mode none
