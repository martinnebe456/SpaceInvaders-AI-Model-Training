Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "Common.ps1")
Activate-Venv
Ensure-MplConfig

$Timesteps = if ($env:TIMESTEPS) { $env:TIMESTEPS } else { "1000000" }
$TrainMaxSteps = if ($env:TRAIN_MAX_STEPS) { $env:TRAIN_MAX_STEPS } else { "20000" }
$NEnvs = if ($env:N_ENVS) { $env:N_ENVS } else { "6" }
$VecEnv = if ($env:VEC_ENV) { $env:VEC_ENV } else { "subproc" }
$TrainDevice = if ($env:TRAIN_DEVICE) { $env:TRAIN_DEVICE } else { "cpu" }
$SilentEvalEpisodes = if ($env:SILENT_EVAL_EPISODES) { $env:SILENT_EVAL_EPISODES } else { "5" }
$SilentEvalMaxSteps = if ($env:SILENT_EVAL_MAX_STEPS) { $env:SILENT_EVAL_MAX_STEPS } else { "20000" }
$HumanEvalEpisodes = if ($env:HUMAN_EVAL_EPISODES) { $env:HUMAN_EVAL_EPISODES } else { "1" }
$HumanEvalMaxSteps = if ($env:HUMAN_EVAL_MAX_STEPS) { $env:HUMAN_EVAL_MAX_STEPS } else { "20000" }
$ModelPath = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { "models/space_invaders_dqn" }
$ReplayBufferPath = if ($env:REPLAY_BUFFER_PATH) { $env:REPLAY_BUFFER_PATH } else { "models/space_invaders_dqn_replay_buffer.pkl" }
$ResetReplayBuffer = if ($env:RESET_REPLAY_BUFFER) { $env:RESET_REPLAY_BUFFER } else { "0" }
$LogDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { "runs/automation" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "training_cycle_$Stamp.log"

Write-Status "Space Invaders RL" "Startuji navazujici trening na $Timesteps kroku."

$trainArguments = @(
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
    $trainArguments += "--reset-replay-buffer"
}

python @trainArguments 2>&1 | Tee-Object -FilePath $LogFile

Write-Status "Space Invaders RL" "Trening dokonceny. Spoustim tichou evaluaci."
python -m space_invaders_rl.evaluate `
    --model-path "$ModelPath.zip" `
    --episodes $SilentEvalEpisodes `
    --max-steps $SilentEvalMaxSteps `
    --render-mode none 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Status "Space Invaders RL" "Ticha evaluace hotova. Spoustim jednu vizualni hru."
python -m space_invaders_rl.evaluate `
    --model-path "$ModelPath.zip" `
    --episodes $HumanEvalEpisodes `
    --max-steps $HumanEvalMaxSteps `
    --render-mode human `
    --full-game 2>&1 | Tee-Object -FilePath $LogFile -Append

Write-Status "Space Invaders RL" "Cely treninkovy cyklus je hotovy."
Write-Host "Log ulozen do $LogFile"
