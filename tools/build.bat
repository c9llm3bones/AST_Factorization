@echo off
setlocal
echo Building Java classes...
mkdir out 2>nul
javac -cp "libs/*" -d out src\build_ast\*.java
if errorlevel 1 (
  echo Compilation failed.
  exit /b 1
)
echo Compilation succeeded.
endlocal
