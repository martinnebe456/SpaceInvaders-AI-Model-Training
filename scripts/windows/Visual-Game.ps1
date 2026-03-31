Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Common.ps1")
# Human playback is useful when you want to inspect the agent qualitatively.
Activate-Venv
Ensure-MplConfig

# These settings can be overridden from the shell for quick experiments.
$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/space_invaders_dqn.zip" }
$Episodes = if ($env:EPISODES) { $env:EPISODES } else { "1" }
$MaxSteps = if ($env:MAX_STEPS) { $env:MAX_STEPS } else { "50000" }

python -m space_invaders_rl.evaluate `
    --model-path $ModelPath `
    --episodes $Episodes `
    --max-steps $MaxSteps `
    --human-max-steps $MaxSteps `
    --render-mode human `
    --full-game
