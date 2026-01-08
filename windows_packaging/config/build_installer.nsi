; NSIS Installer Script for ChemGraph Windows Package
; This script creates a Windows installer using NSIS
; Following NSIS best practices for Windows application distribution

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

;--------------------------------
; Version Information
; Version is passed via /DVERSION= parameter from build script
; Fallback to default if not provided
!ifndef VERSION
    !define VERSION "1.0.0"
!endif

;--------------------------------
; General

; Name and file
Name "ChemGraph"
OutFile "..\ChemGraph-Setup-${VERSION}.exe"
Unicode True

; Version information
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "ChemGraph"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "FileDescription" "ChemGraph - Computational Chemistry Agent"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "CompanyName" "Argonne National Laboratory"
VIAddVersionKey "LegalCopyright" "Copyright (C) 2025"

; Default installation directory
InstallDir "$PROGRAMFILES\ChemGraph"

; Get installation folder from registry if available
InstallDirRegKey HKLM "Software\ChemGraph" "InstallPath"

; Check for existing installation in install section

; Request application privileges for Windows Vista/7/8/10/11
RequestExecutionLevel admin

; Compression
SetCompressor /SOLID lzma
SetCompressorDictSize 32

;--------------------------------
; Variables

Var StartMenuFolder

;--------------------------------
; Interface Settings

!define MUI_ABORTWARNING
!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

;--------------------------------
; Pages

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY

;Start Menu Folder Page Configuration
!define MUI_STARTMENUPAGE_REGISTRY_ROOT "HKCU"
!define MUI_STARTMENUPAGE_REGISTRY_KEY "Software\ChemGraph"
!define MUI_STARTMENUPAGE_REGISTRY_VALUENAME "Start Menu Folder"

!insertmacro MUI_PAGE_STARTMENU Application $StartMenuFolder

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_UNPAGE_FINISH

;--------------------------------
; Languages

!insertmacro MUI_LANGUAGE "English"

;--------------------------------
; Installer Sections

Section "ChemGraph Core" SecCore

    SectionIn RO
    
    ; Check for existing installation
    ReadRegStr $R0 HKLM "Software\ChemGraph" "InstallPath"
    ${If} $R0 != ""
        ${AndIf} $R0 != "$INSTDIR"
        MessageBox MB_YESNO|MB_ICONQUESTION "ChemGraph is already installed at $\n$\n$R0$\n$\nDo you want to uninstall the existing version first?" IDYES uninstall_existing IDNO continue_install
        uninstall_existing:
            ExecWait '"$R0\Uninstall.exe" /S _?=$R0'
            IfErrors 0 continue_install
            MessageBox MB_OK|MB_ICONSTOP "Failed to uninstall existing version. Please uninstall manually and try again."
            Abort
        continue_install:
    ${EndIf}
    
    ; Set output path to the installation directory
    SetOutPath "$INSTDIR"
    
    ; Verify source files exist
    IfFileExists "..\dist\chemgraph\chemgraph.exe" 0 +3
    Goto files_ok
    MessageBox MB_OK|MB_ICONSTOP "Error: Build files not found. Please run build_windows.bat first."
    Abort
    files_ok:
    
    ; Copy all files from dist/chemgraph (relative to windows_packaging root)
    File /r "..\dist\chemgraph\*.*"
    
    ; Store installation folder and version in registry
    WriteRegStr HKLM "Software\ChemGraph" "InstallPath" $INSTDIR
    WriteRegStr HKLM "Software\ChemGraph" "Version" "${VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "DisplayName" "ChemGraph ${VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "DisplayVersion" "${VERSION}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "Publisher" "Argonne National Laboratory"
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph" "NoRepair" 1
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    ; Create Start Menu shortcuts
    !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
        
        ; Create shortcuts
        CreateDirectory "$SMPROGRAMS\$StartMenuFolder"
        CreateShortcut "$SMPROGRAMS\$StartMenuFolder\ChemGraph.lnk" "$INSTDIR\chemgraph.exe"
        CreateShortcut "$SMPROGRAMS\$StartMenuFolder\ChemGraph CLI.lnk" "$INSTDIR\chemgraph.exe" "--help"
        CreateShortcut "$SMPROGRAMS\$StartMenuFolder\Uninstall ChemGraph.lnk" "$INSTDIR\Uninstall.exe"
        
    !insertmacro MUI_STARTMENU_WRITE_END
    
    ; Add to PATH (if EnVar plugin is available)
    IfFileExists "$NSISDIR\Plugins\x86-unicode\EnVar.dll" 0 +3
    EnVar::SetHKLM
    EnVar::AddValue "PATH" "$INSTDIR"
    
    ; Create desktop shortcut
    CreateShortcut "$DESKTOP\ChemGraph.lnk" "$INSTDIR\chemgraph.exe"
    
SectionEnd

Section "Documentation" SecDocs

    SetOutPath "$INSTDIR\docs"
    File /r "..\..\docs\*.*"
    
    ; Create documentation shortcut
    !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
        CreateShortcut "$SMPROGRAMS\$StartMenuFolder\Documentation.lnk" "$INSTDIR\docs\index.html"
    !insertmacro MUI_STARTMENU_WRITE_END

SectionEnd

Section "Example Notebooks" SecExamples

    SetOutPath "$INSTDIR\notebooks"
    File /r "..\..\notebooks\*.ipynb"
    
    ; Create examples shortcut
    !insertmacro MUI_STARTMENU_WRITE_BEGIN Application
        CreateShortcut "$SMPROGRAMS\$StartMenuFolder\Example Notebooks.lnk" "$INSTDIR\notebooks"
    !insertmacro MUI_STARTMENU_WRITE_END

SectionEnd

;--------------------------------
; Descriptions

; Language strings
LangString DESC_SecCore ${LANG_ENGLISH} "Core ChemGraph application and all required dependencies."
LangString DESC_SecDocs ${LANG_ENGLISH} "Install documentation files."
LangString DESC_SecExamples ${LANG_ENGLISH} "Install example Jupyter notebooks."

; Assign language strings to sections
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} $(DESC_SecCore)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDocs} $(DESC_SecDocs)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecExamples} $(DESC_SecExamples)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Uninstaller Section

Section "Uninstall"

    ; Remove files and uninstaller
    RMDir /r "$INSTDIR"
    
    ; Remove Start Menu shortcuts
    !insertmacro MUI_STARTMENU_GETFOLDER Application $StartMenuFolder
    Delete "$SMPROGRAMS\$StartMenuFolder\ChemGraph.lnk"
    Delete "$SMPROGRAMS\$StartMenuFolder\ChemGraph CLI.lnk"
    Delete "$SMPROGRAMS\$StartMenuFolder\Uninstall ChemGraph.lnk"
    Delete "$SMPROGRAMS\$StartMenuFolder\Documentation.lnk"
    Delete "$SMPROGRAMS\$StartMenuFolder\Example Notebooks.lnk"
    RMDir "$SMPROGRAMS\$StartMenuFolder"
    
    ; Remove desktop shortcut
    Delete "$DESKTOP\ChemGraph.lnk"
    
    ; Remove from PATH (if EnVar plugin is available)
    IfFileExists "$NSISDIR\Plugins\x86-unicode\EnVar.dll" 0 +3
    EnVar::SetHKLM
    EnVar::DeleteValue "PATH" "$INSTDIR"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\ChemGraph"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ChemGraph"

SectionEnd

