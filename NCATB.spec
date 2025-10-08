# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import certifi


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

certifi_datas = [(certifi.where(), 'certifi')]


a = Analysis(
    ['main.py'],
    pathex=[str(BASE_DIR)],
    binaries=[],
    datas=certifi_datas,
    hiddenimports=['pandas_ta', 'ccxt', 'PyQt6.sip'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(BASE_DIR / 'tls_runtime_hook.py')],
    excludes=['api_keys.txt'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Neam Coin Auto Trading Bot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None,
    entitlements_file=None,
    icon=str(BASE_DIR / 'icon.icns'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Neam Coin Auto Trading Bot',
)

app = BUNDLE(
    coll,
    name='Neam Coin Auto Trading Bot.app',
    icon=str(BASE_DIR / 'icon.icns'),
    bundle_identifier=None,
)
