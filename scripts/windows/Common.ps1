$Script:RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

function Enter-RepoRoot {
    Set-Location $Script:RepoRoot
}

function Activate-Venv {
    Enter-RepoRoot
    $activatePath = Join-Path $Script:RepoRoot ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path $activatePath)) {
        throw "Virtualni prostredi .venv nebylo nalezeno."
    }
    . $activatePath
}

function Ensure-MplConfig {
    $mplConfig = if ($env:MPLCONFIGDIR) { $env:MPLCONFIGDIR } else { Join-Path $Script:RepoRoot ".mplconfig" }
    New-Item -ItemType Directory -Force -Path $mplConfig | Out-Null
    $env:MPLCONFIGDIR = $mplConfig
}

function Write-Status {
    param(
        [string]$Title,
        [string]$Message
    )
    $time = Get-Date -Format "HH:mm:ss"
    Write-Host "[$time] $Title - $Message"
}
