# PyInstaller spec for the NSD Tread Depth Estimator desktop app.
# Build with:  pyinstaller nsd_gui.spec --noconfirm
#
# Ships as a folder (--onedir) rather than a single .exe on purpose: the
# bundle already contains ~220MB of model checkpoints plus the torch/cv2
# runtime, so a --onefile build would re-extract all of that into a temp
# dir on every single launch, making startup noticeably slower for no
# real benefit here. Zip the whole `dist/NSD_Tread_Depth_Estimator`
# folder to hand it to someone else; they run the .exe inside it.

import os
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None
PROJECT_DIR = os.path.abspath(SPECPATH)

datas = [
    (
        os.path.join(PROJECT_DIR, "checkpoint_1600_data_attention_excluded_2to9"),
        "checkpoint_1600_data_attention_excluded_2to9",
    ),
]
datas += collect_data_files("ttkbootstrap")

a = Analysis(
    ["gui_app.py"],
    pathex=[PROJECT_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "notebook", "IPython"],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="NSD_Tread_Depth_Estimator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="NSD_Tread_Depth_Estimator",
)
