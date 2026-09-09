@echo off
title PUPPET EXHIBITION - BUILD 5.3
cd /d "%~dp0"
cls
echo ============================================================
echo PUPPET EXHIBITION BUILD 5.3 - 15 FPS / VISITOR HANDOVER
echo ============================================================
echo.
echo Water break: after 10 separate visitor sessions.
echo THANK YOU: 5 seconds, then automatic reset.
echo Different visitor appearance: automatic new round.
echo Visitor archive IDs begin at N.00001 and continue after restart.
echo Arduino firmware remains BUILD 4-compatible on servo pin D9.
echo.
py PUPPET_EXHIBITION_BUILD.py
echo.
echo Program closed. Read any error shown above.
pause
