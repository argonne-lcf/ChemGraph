# Windows Packaging Directory Structure

All Windows-related files are organized in the `windows_packaging/` directory with the following structure:

## Directory Organization

```
windows_packaging/
│
├── README.md                    # Main entry point - start here
│
├── scripts/                     # Build and utility scripts
│   ├── build_windows.bat        # Builds standalone executable
│   ├── build_installer.bat       # Creates Windows installer
│   └── launcher.bat             # Alternative launcher
│
├── config/                      # Configuration files
│   ├── chemgraph.spec           # PyInstaller specification
│   ├── build_installer.nsi      # NSIS installer script
│   └── requirements-build.txt   # Build dependencies
│
├── docs/                        # Documentation
│   ├── README.md                # Comprehensive guide
│   ├── QUICKSTART.md            # Quick reference
│   └── PACKAGING_SUMMARY.md     # Technical overview
│
├── dist/                        # Build output (gitignored)
│   └── chemgraph/
│       └── chemgraph.exe
│
└── build/                       # PyInstaller build files (gitignored)
```

## File Categories

### Scripts (`scripts/`)
All executable batch scripts for building and running:
- **build_windows.bat**: Main build script for creating executable
- **build_installer.bat**: Creates the Windows installer
- **launcher.bat**: Alternative way to run ChemGraph (requires Python)

### Configuration (`config/`)
All configuration and specification files:
- **chemgraph.spec**: PyInstaller configuration
- **build_installer.nsi**: NSIS installer script
- **requirements-build.txt**: Additional build dependencies

### Documentation (`docs/`)
All documentation files:
- **README.md**: Comprehensive packaging guide
- **QUICKSTART.md**: Quick start reference
- **PACKAGING_SUMMARY.md**: Technical details

### Build Outputs
Created during build process (gitignored):
- **dist/**: Contains the built executable
- **build/**: PyInstaller temporary files

## Usage

1. **Start here**: Read `README.md` in the root of `windows_packaging/`
2. **Quick build**: Run `scripts\build_windows.bat`
3. **Create installer**: Run `scripts\build_installer.bat`
4. **Detailed info**: See files in `docs/` directory

## Benefits of This Organization

- ✅ **Clear separation**: Scripts, configs, and docs are separated
- ✅ **Easy navigation**: Find files quickly by category
- ✅ **Clean root**: Main README at root level
- ✅ **Maintainable**: Easy to add new files in appropriate locations
- ✅ **Professional**: Follows standard project organization practices
