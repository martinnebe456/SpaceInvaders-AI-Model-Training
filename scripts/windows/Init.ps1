Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Bootstrap = Join-Path $RepoRoot "scripts\bootstrap\init_env.py"

function Test-PyVersion {
    param([string]$VersionTag)
    & py $VersionTag -c "import sys; print(sys.version)" *> $null
    return ($LASTEXITCODE -eq 0)
}

Set-Location $RepoRoot

if ($env:PYTHON_BIN) {
    & $env:PYTHON_BIN $Bootstrap --platform windows @args
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    if (Test-PyVersion "-3.12") {
        & py -3.12 $Bootstrap --platform windows @args
        exit $LASTEXITCODE
    }
    if (Test-PyVersion "-3.13") {
        & py -3.13 $Bootstrap --platform windows @args
        exit $LASTEXITCODE
    }
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $Bootstrap --platform windows @args
    exit $LASTEXITCODE
}

throw "Python 3.12 nebo 3.13 nebyl nalezen. Nainstaluj ho a spust init znovu."
