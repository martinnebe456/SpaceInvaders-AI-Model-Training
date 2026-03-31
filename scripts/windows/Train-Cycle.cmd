@echo off
REM Thin CMD wrapper that forwards to the PowerShell training-cycle script.
powershell -ExecutionPolicy Bypass -File "%~dp0Train-Cycle.ps1" %*
