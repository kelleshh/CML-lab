@echo off
chcp 65001 >nul
call "%~dp0start.bat" --doctor %*
set "LAB_STATUS=%ERRORLEVEL%"
if "%LAB_STATUS%"=="0" pause
exit /b %LAB_STATUS%
