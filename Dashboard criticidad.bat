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