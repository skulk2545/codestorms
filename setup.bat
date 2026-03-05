@echo off
echo [*] Starting HandScript-to-Text Setup...
echo [*] Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [*] Verifying installation...
python check_env.py

echo.
echo [!] REMINDER: Ensure Poppler is installed and in your PATH.
echo [!] If 'check_env.py' shows Poppler is missing:
echo [!] 1. Download from https://github.com/oschwartz10612/poppler-windows/releases/
echo [!] 2. Extract and add the 'bin' folder to your System ENV PATH.
echo.
pause
