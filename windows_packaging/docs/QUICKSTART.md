# Quick Start Guide: Building Windows Package

This is a quick reference guide for building the Windows standalone package. For detailed information, see [README.md](README.md).

## Prerequisites Checklist

- [ ] Python 3.10+ installed and in PATH
- [ ] NSIS installed (for installer creation)
- [ ] Git repository cloned
- [ ] All project dependencies can be installed

## Build Steps

### 1. Build Executable (5-15 minutes)

**Option A: From windows_packaging directory (Recommended)**
```batch
cd windows_packaging
scripts\build_windows.bat
```

**Option B: From project root**
```batch
cd path\to\ChemGraph
windows_packaging\scripts\build_windows.bat
```

**What it does:**
- Installs PyInstaller
- Installs ChemGraph and dependencies
- Creates standalone executable
- Automatically navigates to correct directories

**Output:** `windows_packaging\dist\chemgraph\chemgraph.exe`

**Note:** The script handles path navigation automatically, so you can run it from either location.

### 2. Test Executable

```batch
windows_packaging\dist\chemgraph\chemgraph.exe --help
```

### 3. Build Installer (1-2 minutes)

**From windows_packaging directory:**
```batch
cd windows_packaging
scripts\build_installer.bat
```

**Output:** `windows_packaging\ChemGraph-Setup-{VERSION}.exe`

**Note:** Make sure you've completed Step 1 (build executable) first!

## Distribution

The `ChemGraph-Setup.exe` file can be distributed to Windows users. They can:
1. Download the installer
2. Run it (may need admin rights)
3. Follow the installation wizard
4. Use `chemgraph` command from anywhere

## Common Issues

**"Python not found"**
- Add Python to PATH or use full path to python.exe

**"NSIS not found"**
- Install NSIS and add to PATH, or skip installer step

**Build takes too long**
- Normal for first build (downloading dependencies)
- Subsequent builds are faster

**Executable is large (>500MB)**
- Expected - includes all scientific libraries
- Compression is enabled to reduce size

## Next Steps

- Test the installer on a clean Windows machine
- Set up API keys for testing
- Create user documentation

