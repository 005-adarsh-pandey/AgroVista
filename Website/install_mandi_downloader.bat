@echo off
echo Installing required packages for Mandi Auto Downloader...

pip install schedule requests pandas pywin32

echo.
echo Packages installed successfully!
echo.
echo To install as Windows Service:
echo python mandi_service.py install
echo.
echo To start the service:
echo python mandi_service.py start
echo.
echo To stop the service:
echo python mandi_service.py stop
echo.
echo To remove the service:
echo python mandi_service.py remove
echo.
echo To test single download:
echo python mandi_auto_downloader.py
echo.
pause
