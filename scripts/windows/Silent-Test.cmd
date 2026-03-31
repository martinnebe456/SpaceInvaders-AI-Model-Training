@echo off
REM Thin CMD wrapper that forwards to the PowerShell silent evaluation script.
powershell -ExecutionPolicy Bypass -File "%~dp0Silent-Test.ps1" %*
