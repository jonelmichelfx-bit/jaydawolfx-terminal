@echo off
REM ================================================================
REM JayDaWolfX Terminal — Build Desktop .exe
REM Double-click this file to build your standalone .exe
REM Output: dist\JayDaWolfX.exe
REM ================================================================

echo.
echo  ================================================================
echo   JayDaWolfX Terminal — Building Desktop App
echo   This takes 3-7 minutes. Do not close this window.
echo  ================================================================
echo.

REM Step 1 — Install desktop dependencies
echo  [1/5] Installing flaskwebgui and pyinstaller...
pip install flaskwebgui pyinstaller --quiet
echo  Done.
echo.

REM Step 2 — Clean old build files
echo  [2/5] Cleaning old build files...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist
if exist JayDaWolfX.spec del /q JayDaWolfX.spec
echo  Done.
echo.

REM Step 3 — Build the .exe
echo  [3/5] Building .exe with PyInstaller...
echo  (this is the slow step — grab a coffee)
echo.

pyinstaller ^
  --name "JayDaWolfX" ^
  --onefile ^
  --windowed ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --hidden-import flask ^
  --hidden-import flask_sqlalchemy ^
  --hidden-import flask_login ^
  --hidden-import flask.cli ^
  --hidden-import flaskwebgui ^
  --hidden-import anthropic ^
  --hidden-import yfinance ^
  --hidden-import stripe ^
  --hidden-import numpy ^
  --hidden-import scipy ^
  --hidden-import scipy.stats ^
  --hidden-import dotenv ^
  --hidden-import python_dotenv ^
  --hidden-import sqlalchemy ^
  --hidden-import sqlalchemy.dialects.sqlite ^
  --hidden-import requests ^
  --hidden-import concurrent.futures ^
  --hidden-import zoneinfo ^
  --hidden-import auth ^
  --hidden-import models ^
  --hidden-import decorators ^
  --hidden-import payments ^
  --hidden-import scanner ^
  --hidden-import forex ^
  --hidden-import wolf_agent ^
  --hidden-import psutil ^
  launcher.py

echo.

REM Step 4 — Copy .env into dist folder
echo  [4/5] Copying .env into dist folder...
if exist .env (
    copy .env dist\.env >nul
    echo  .env copied successfully.
) else (
    echo  WARNING: No .env file found in this folder.
    echo  Create one from .env.template before sharing the .exe.
)
echo.

REM Step 5 — Done
echo  [5/5] Build complete!
echo.
echo  ================================================================
echo   Your .exe is ready at:  dist\JayDaWolfX.exe
echo.
echo   To run it:  double-click dist\JayDaWolfX.exe
echo.
echo   To share it with someone:
echo     1. Zip the entire dist\ folder
echo     2. They unzip and double-click JayDaWolfX.exe
echo     3. No Python, no browser, no install needed
echo  ================================================================
echo.
pause
