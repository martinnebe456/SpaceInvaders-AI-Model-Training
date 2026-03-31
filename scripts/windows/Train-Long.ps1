Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Common.ps1")
Activate-Venv
Ensure-MplConfig

$Timesteps = if ($env:TIMESTEPS) { $env:TIMESTEPS } else { "100000000" }
$TrainMaxSteps = if ($env:TRAIN_MAX_STEPS) { $env:TRAIN_MAX_STEPS } else { "20000" }
$NEnvs = if ($env:N_ENVS) { $env:N_ENVS } else { "10" }
$VecEnv = if ($env:VEC_ENV) { $env:VEC_ENV } else { "subproc" }
$TrainDevice = if ($env:TRAIN_DEVICE) { $env:TRAIN_DEVICE } else { "cpu" }
$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/space_invaders_dqn" }
$ReplayBufferPath = if ($env:REPLAY_BUFFER_PATH) { $env:REPLAY_BUFFER_PATH } else { "models/space_invaders_dqn_replay_buffer.pkl" }
$ResetReplayBuffer = if ($env:RESET_REPLAY_BUFFER) { $env:RESET_REPLAY_BUFFER } else { "0" }
$LogDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { "runs/automation" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "long_training_$Stamp.log"

Write-Status "Space Invaders RL" "Startuji dlouhy navazujici trening na $Timesteps kroku."

$arguments = @(
    "-m", "space_invaders_rl.train",
    "--timesteps", $Timesteps,
    "--max-steps", $TrainMaxSteps,
    "--n-envs", $NEnvs,
    "--vec-env", $VecEnv,
    "--device", $TrainDevice,
    "--model-path", $ModelPath,
    "--replay-buffer-path", $ReplayBufferPath,
    "--skip-check-env"
)

if ($ResetReplayBuffer -eq "1") {
    $arguments += "--reset-replay-buffer"
}

python @arguments 2>&1 | Tee-Object -FilePath $LogFile

Write-Status "Space Invaders RL" "Dlouhy trening je hotovy."
Write-Host "Log ulozen do $LogFile"
