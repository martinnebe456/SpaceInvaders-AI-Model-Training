$Script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

# Move every wrapper back to the repo root so relative paths behave consistently.
function Enter-RepoRoot {
    Set-Location $Script:RepoRoot
}

# Activate the project-local virtual environment or fail with a clear message.
function Activate-Venv {
    Enter-RepoRoot
    $activatePath = Join-Path $Script:RepoRoot ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $activatePath)) {
        throw "Virtual environment .venv was not found."
    }
    . $activatePath
}

# Keep Matplotlib cache files inside the repo so runs behave predictably.
function Ensure-MplConfig {
    $mplConfig = if ($env:MPLCONFIGDIR) { $env:MPLCONFIGDIR } else { Join-Path $Script:RepoRoot ".mplconfig" }
    New-Item -ItemType Directory -Force -Path $mplConfig | Out-Null
    $env:MPLCONFIGDIR = $mplConfig
}

# Emit consistent timestamped status lines for logs and console output.
function Write-Status {
    param(
        [string]$Title,
        [string]$Message
    )
    $time = Get-Date -Format "HH:mm:ss"
    Write-Host "[$time] $Title - $Message"
}
