@echo off
REM Thin CMD wrapper for users who prefer Command Prompt over PowerShell.
powershell -ExecutionPolicy Bypass -File "%~dp0Benchmark.ps1" %*
