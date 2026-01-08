# Running Build Scripts on Windows

## Step-by-Step Instructions

### 1. Open Command Prompt

You can open Command Prompt in several ways:

**Method 1: Using Run Dialog**
- Press `Windows Key + R`
- Type `cmd`
- Press `Enter`

**Method 2: Using Start Menu**
- Click the Start button
- Type "Command Prompt" or "cmd"
- Click on "Command Prompt"

**Method 3: From File Explorer**
- Navigate to the `windows_packaging` folder in File Explorer
- Click in the address bar, type `cmd`, press Enter
- This opens Command Prompt in that directory

### 2. Navigate to Your Project

**If Command Prompt opened in a different location:**

```batch
cd C:\Users\YourName\Documents\ChemGraph
```

(Replace with your actual path to the ChemGraph project)

**To find your path:**
- Open File Explorer
- Navigate to the ChemGraph folder
- Click in the address bar to see the full path
- Copy that path and use it in the `cd` command

### 3. Run the Build Script

**Option A: From windows_packaging directory (Recommended)**

```batch
cd windows_packaging
scripts\build_windows.bat
```

**Option B: From project root**

```batch
windows_packaging\scripts\build_windows.bat
```

### 4. Wait for Build to Complete

The script will:
1. Check Python version
2. Install/upgrade PyInstaller
3. Install ChemGraph and dependencies
4. Build the executable

This takes 5-15 minutes the first time (downloading dependencies).

### 5. Verify the Build

After the build completes, you should see:
```
Build completed successfully!
Executable location: windows_packaging\dist\chemgraph\chemgraph.exe
```

### 6. Test the Executable (Optional)

```batch
windows_packaging\dist\chemgraph\chemgraph.exe --help
```

### 7. Build the Installer

```batch
cd windows_packaging
scripts\build_installer.bat
```

## Common Issues

### "Python is not recognized"

**Problem:** Python is not in your PATH.

**Solution:**
1. Reinstall Python and check "Add Python to PATH" during installation
2. Or use the full path to Python:
   ```batch
   C:\Python310\python.exe -m pip install -e .
   ```

### "The system cannot find the path specified"

**Problem:** You're not in the correct directory.

**Solution:**
1. Check your current directory: `cd`
2. Navigate to the correct location:
   ```batch
   cd C:\path\to\ChemGraph\windows_packaging
   ```

### "NSIS not found" (for installer step)

**Problem:** NSIS is not installed or not in PATH.

**Solution:**
1. Download NSIS from: https://nsis.sourceforge.io/Download
2. Install it
3. Add NSIS to PATH, or use full path to `makensis.exe`

## Tips

- **Keep Command Prompt open** during the build process
- **Don't close** the window while building
- **Read error messages** carefully - they often tell you what's wrong
- **Check Python version** first: `python --version` (should be 3.10+)

## Example Session

Here's what a complete session looks like:

```batch
C:\Users\YourName>cd Documents\ChemGraph
C:\Users\YourName\Documents\ChemGraph>cd windows_packaging
C:\Users\YourName\Documents\ChemGraph\windows_packaging>scripts\build_windows.bat

========================================
ChemGraph Windows Build Script
========================================

[1/6] Checking Python version...
Python version: 3.11.5
...
[6/6] Verifying build...

========================================
Build completed successfully!
========================================

C:\Users\YourName\Documents\ChemGraph\windows_packaging>scripts\build_installer.bat
...
```

## Need Help?

If you encounter issues:
1. Check the error message in Command Prompt
2. See [QUICKSTART.md](docs/QUICKSTART.md) for troubleshooting
3. Review [README.md](README.md) for detailed documentation
