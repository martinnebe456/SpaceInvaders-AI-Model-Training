@echo off
REM Thin CMD wrapper that forwards to the PowerShell bootstrap script.
powershell -ExecutionPolicy Bypass -File "%~dp0Init.ps1" %*
