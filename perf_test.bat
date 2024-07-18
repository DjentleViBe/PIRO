rem Copy assets from ./dependencies/assets to ./bin/
xcopy /E /I /Y .\dependencies\assets .\bin\assets\

rem Create the 'logs' directory if it doesn't exist
if not exist logs mkdir logs

.\bin\GIRO.exe setup.ini
.\bin\GIRO.exe setup_1.ini
.\bin\GIRO.exe setup_2.ini
.\bin\GIRO.exe setup_3.ini
