@echo off
setlocal
if "%1"=="" (
  set SRC_DIR=java_src
) else (
  set SRC_DIR=%1
)
if "%2"=="" (
  set OUT_DIR=method_jsons
) else (
  set OUT_DIR=%2
)
mkdir "%OUT_DIR%" 2>nul

REM Choose which main class to run: ASTPerMethodBuilder (preferred) or ASTBuilder
set MAIN_CLASS=build_ast.ASTPerMethodBuilder

for /R "%SRC_DIR%" %%f in (*.java) do (
  echo Processing %%f
  java -cp "out;libs/*" %MAIN_CLASS% "%%f" "%OUT_DIR%"
  if errorlevel 1 (
    echo ERROR processing %%f
  )
)
echo Done processing all java files.
endlocal
