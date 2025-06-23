@ECHO OFF

REM Command file for MkDocs documentation

SET MKDOCS=mkdocs
SET DOCS_DIR=docs
SET SITE_DIR=site

IF "%1" == "" GOTO help

%MKDOCS% build --config-file "%DOCS_DIR%/mkdocs.yml" --site-dir "%SITE_DIR%"
GOTO end

:help
ECHO Usage: %0 [command]
ECHO.
ECHO Available commands:
ECHO   build     Build the MkDocs documentation
ECHO   serve     Serve the MkDocs documentation locally
ECHO   deploy    Deploy the MkDocs documentation (e.g., to GitHub Pages)

:end
