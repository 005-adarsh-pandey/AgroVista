@echo off
echo ===============================================
echo   AgroVista Mandi Auto-Downloader Setup
echo ===============================================
echo.

echo Installing required Python packages...
pip install schedule requests pandas

echo.
echo ===============================================
echo Setup completed! 
echo ===============================================
echo.
echo Available options:
echo.
echo 1. Test single download:
echo    python simple_mandi_downloader.py --once
echo.
echo 2. Run continuous downloader (recommended):
echo    python simple_mandi_downloader.py
echo.
echo 3. Run Flask app with auto-downloader:
echo    python app.py
echo.
echo ===============================================
echo Market Hours: 6 AM to 6 PM (downloads every hour)
echo Data Location: static/data/mandi_prices_*.csv
echo Log File: mandi_downloader.log
echo ===============================================
echo.
echo Press any key to test a single download...
pause > nul

echo.
echo Testing single download...
python simple_mandi_downloader.py --once

echo.
echo Test completed! Check the static/data folder for new files.
echo.
echo To start continuous monitoring, run:
echo python simple_mandi_downloader.py
echo.
pause
