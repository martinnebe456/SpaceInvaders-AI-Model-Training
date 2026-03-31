@echo off
REM Thin CMD wrapper that forwards to the PowerShell visual-playback script.
powershell -ExecutionPolicy Bypass -File "%~dp0Visual-Game.ps1" %*
