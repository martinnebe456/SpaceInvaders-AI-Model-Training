@echo off
REM Thin CMD wrapper that forwards to the PowerShell long-training script.
powershell -ExecutionPolicy Bypass -File "%~dp0Train-Long.ps1" %*
