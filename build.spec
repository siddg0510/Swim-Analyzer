# PyInstaller spec.
#
# Build on the SAME OS you're targeting — PyInstaller does not
# cross-compile. Build on Windows for a .exe, on macOS for a .app.
#
#   pip install -r requirements.txt
#   (download models/pose_landmarker.task first — see models/DOWNLOAD_MODEL.md)
#   pyinstaller build.spec
#
# Output lands in dist/SwimRaceAnalyzer/ (onedir) — onedir is used
# instead of onefile because mediapipe + opencv + PySide6 unpack a large
# number of native libraries, and onefile's every-launch unpack-to-temp
# behaviour makes startup noticeably slower for a bundle this size with
# no real benefit for a desktop coaching tool.

import imageio_ffmpeg

block_cipher = None

ffmpeg_binary = imageio_ffmpeg.get_ffmpeg_exe()

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[(ffmpeg_binary, '.')],
    datas=[
        ('models/pose_landmarker.task', 'models'),
    ],
    hiddenimports=[
        'mediapipe.tasks.python.vision',
        'mediapipe.tasks.python.core',
        'matplotlib.backends.backend_qtagg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SwimRaceAnalyzer',
    debug=False,
    strip=False,
    upx=True,
    console=False,   # no terminal window on Windows
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='SwimRaceAnalyzer',
)

# macOS only: wrap into a proper .app bundle.
app = BUNDLE(
    coll,
    name='SwimRaceAnalyzer.app',
    icon=None,
    bundle_identifier='com.local.swimraceanalyzer',
)
