@echo off
setlocal

REM Dossier du projet = dossier du .bat
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

REM Dossier de l'environnement virtuel
set "VENV_DIR=%PROJECT_DIR%.venv"

REM 1) Créer le venv s'il n'existe pas
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creation de l'environnement virtuel...
    py -3.11 -m venv "%VENV_DIR%" 2>nul || py -3 -m venv "%VENV_DIR%" 2>nul || python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERREUR] Impossible de creer l'environnement virtuel.
        exit /b 1
    )
)

REM 2) Activer le venv
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel.
    pause
    exit /b 1
)

REM 3) Installer les dependances si requirements.txt existe
if exist "%PROJECT_DIR%requirements.txt" (
    echo [INFO] Installation des dependances...
    pip install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 (
        echo [ERREUR] Echec de l'installation des dependances.
        pause
        exit /b 1
    )
)

REM 4) Lancer Streamlit
echo [INFO] Lancement de l'application Streamlit...
python -m streamlit run app.py

echo.
echo [INFO] Fin de l'application. Appuie sur une touche pour fermer.
pause

endlocal
