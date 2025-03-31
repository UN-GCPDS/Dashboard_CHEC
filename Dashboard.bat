::[Bat To Exe Converter]
::
::YAwzoRdxOk+EWAjk
::fBw5plQjdCyDJGyX8VAjFBdBTwWRAES0A5EO4f7+r6fHl0MUQucta4bf27DODuEQ40rqdJpt0n8au8QAAxZadxOXXix5jk1ykle5E8ifpgLkRFyG9XcUGnZ8hWzRni8EQ/tNuf8K0C+wskT8kMU=
::YAwzuBVtJxjWCl3EqQJgSA==
::ZR4luwNxJguZRRnk
::Yhs/ulQjdF+5
::cxAkpRVqdFKZSjk=
::cBs/ulQjdF+5
::ZR41oxFsdFKZSDk=
::eBoioBt6dFKZSDk=
::cRo6pxp7LAbNWATEpCI=
::egkzugNsPRvcWATEpCI=
::dAsiuh18IRvcCxnZtBJQ
::cRYluBh/LU+EWAjk
::YxY4rhs+aU+JeA==
::cxY6rQJ7JhzQF1fEqQJQ
::ZQ05rAF9IBncCkqN+0xwdVs0
::ZQ05rAF9IAHYFVzEqQJQ
::eg0/rx1wNQPfEVWB+kM9LVsJDGQ=
::fBEirQZwNQPfEVWB+kM9LVsJDGQ=
::cRolqwZ3JBvQF1fEqQJQ
::dhA7uBVwLU+EWDk=
::YQ03rBFzNR3SWATElA==
::dhAmsQZ3MwfNWATElA==
::ZQ0/vhVqMQ3MEVWAtB9wSA==
::Zg8zqx1/OA3MEVWAtB9wSA==
::dhA7pRFwIByZRRnk
::Zh4grVQjdCyDJGyX8VAjFBdBTwWRAES0A5EO4f7+r6fHl0MUQucta4bf27DODuEQ40rqdJpt0n8au8QAAxZadxOXXix5jk1ykle5E8ifpgLkRFyG9XcUGnZ8hWzRni8EQ/tNuf8n0jO2/kL+jaFe1GD6Pg==
::YB416Ek+ZG8=
::
::
::978f952a14a936cc963da21a135fa983
@echo off
ECHO Iniciando la secuencia de activación...

REM Cambiar al directorio del script
CD /D "%~dp0"

REM Activar el entorno virtual (DESCOMENTAR LA LÍNEA QUE NECESITES)


SET VENV_PATH=C:\Users\lucas\OneDrive - Universidad Nacional de Colombia\PC-GCPDS\Documentos\venv
REM Si el venv está un nivel arriba
CALL "%VENV_PATH%\Scripts\activate.bat"


IF %ERRORLEVEL% NEQ 0 (
    ECHO Error al activar el entorno virtual
    PAUSE
    EXIT /B 1
)

REM Ejecutar el script de Python
ECHO Ejecutando script de Python...
"%VENV_PATH%\Scripts\python.exe" "main.py"
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error al ejecutar el script de Python
    PAUSE
    EXIT /B 1
)

ECHO Proceso completado exitosamente!
PAUSE