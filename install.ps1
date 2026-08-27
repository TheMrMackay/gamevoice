<#
.SYNOPSIS
    Install GameVoice.

.DESCRIPTION
    Sets up an isolated Python environment, installs the dependencies, fetches a
    starter set of neural voices, builds the icon and creates shortcuts.

    Everything lands inside this folder. Nothing is written to the registry, no
    system Python is modified, and uninstalling is deleting the folder plus the
    two shortcuts.

.PARAMETER NoVoices
    Skip the ~260 MB voice download. GameVoice falls back to the voices built
    into Windows, and voices can be added later from the Voices tab.

.PARAMETER NoShortcuts
    Do not create Desktop or Start Menu shortcuts.

.PARAMETER Python
    Path to a specific python.exe. By default the newest 3.10+ found is used.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File install.ps1 -NoVoices
#>
[CmdletBinding()]
param(
    [switch]$NoVoices,
    [switch]$NoShortcuts,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

$MinimumPython = [version]'3.10'
# Two single-speaker voices for a guaranteed male and female, plus the
# multi-speaker model that supplies a distinct voice per character.
$StarterVoices = @(
    'en_US-ryan-high',
    'en_US-hfc_female-medium',
    'en_US-libritts_r-medium'
)

function Write-Step  { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok    { param([string]$Text) Write-Host "    $Text" -ForegroundColor Green }
function Write-Note  { param([string]$Text) Write-Host "    $Text" -ForegroundColor DarkGray }

function Get-PythonCandidates {
    $found = @()
    # The launcher knows about every install, including ones off PATH.
    try {
        foreach ($line in & py -0p 2>$null) {
            if ($line -match '^\s*-V:(\d+\.\d+)[^\s]*\s+\*?\s*(.+\.exe)\s*$') {
                $found += [pscustomobject]@{ Version = [version]$Matches[1]; Path = $Matches[2].Trim() }
            }
        }
    } catch { }

    foreach ($name in @('python', 'python3')) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        try {
            $raw = & $command.Source -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>$null
            if ($raw) { $found += [pscustomobject]@{ Version = [version]$raw; Path = $command.Source } }
        } catch { }
    }

    $found | Where-Object { $_.Version -ge $MinimumPython } |
        Sort-Object -Property Version -Descending -Unique
}

function Assert-Windows {
    if ($env:OS -ne 'Windows_NT') {
        throw 'GameVoice is Windows only: it uses the Windows OCR engine and Win32 window APIs.'
    }
    $build = [Environment]::OSVersion.Version.Build
    if ($build -lt 17763) {
        Write-Warning "Windows build $build is older than tested (1809 / 17763). The OCR engine may be unavailable."
    }
}

# --- 1. Python ---------------------------------------------------------------

Write-Step 'Looking for Python'
Assert-Windows

if ($Python) {
    if (-not (Test-Path $Python)) { throw "No python.exe at $Python" }
    $chosen = $Python
} else {
    $candidates = @(Get-PythonCandidates)
    if (-not $candidates) {
        throw @"
No Python $MinimumPython or newer found.

Install it from https://www.python.org/downloads/ (tick "Add python.exe to PATH"),
then run this script again.
"@
    }
    $chosen = $candidates[0].Path
}
$reported = & $chosen --version
Write-Ok "$reported at $chosen"

# --- 2. Environment ----------------------------------------------------------

Write-Step 'Creating the environment'
if (Test-Path $venvPython) {
    Write-Ok 'Already present, reusing it.'
} else {
    & $chosen -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment.' }
    Write-Ok "Created $venv"
}

& $venvPython -m pip install --upgrade pip --quiet --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }

# --- 3. Dependencies ---------------------------------------------------------

Write-Step 'Installing dependencies'
Write-Note 'Around 250 MB. This is the slow part.'
& $venvPython -m pip install -e $root --disable-pip-version-check
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. The output above says why.' }
Write-Ok 'Installed.'

# --- 4. Voices ---------------------------------------------------------------

if ($NoVoices) {
    Write-Step 'Skipping voices'
    Write-Note 'The Windows built-in voices will be used. Add neural voices later from the Voices tab.'
} else {
    Write-Step 'Downloading voices'
    Write-Note "$($StarterVoices.Count) models, about 260 MB, downloaded once and used offline."
    $voicesDir = Join-Path $root 'voices'
    New-Item -ItemType Directory -Force $voicesDir | Out-Null

    foreach ($voice in $StarterVoices) {
        if (Test-Path (Join-Path $voicesDir "$voice.onnx")) {
            Write-Ok "$voice already present"
            continue
        }
        Write-Note "fetching $voice ..."
        & $venvPython -c @"
import sys
from pathlib import Path
from piper.download_voices import download_voice
try:
    download_voice('$voice', Path(r'$voicesDir'))
except Exception as exc:
    print(f'  failed: {exc}', file=sys.stderr)
    raise SystemExit(1)
"@
        if ($LASTEXITCODE -ne 0) { throw "Could not download $voice. Check the network and run this again." }
        Write-Ok "$voice"
    }
}

# --- 5. Voice gender table ---------------------------------------------------

$table = Join-Path $root 'data\voice_table.json'
if (-not $NoVoices) {
    Write-Step 'Preparing voices'
    $needsMeasuring = $true
    if (Test-Path $table) {
        # The shipped table already covers the starter set; only re-measure if
        # a voice is installed that it does not know about.
        $known = & $venvPython -c @"
import json, pathlib
table = json.loads(pathlib.Path(r'$table').read_text(encoding='utf-8'))
have = {p.stem for p in pathlib.Path(r'$root\voices').glob('*.onnx')}
print('yes' if have and have <= set(table.get('models', {})) else 'no')
"@
        if ($known -eq 'yes') { $needsMeasuring = $false }
    }

    if ($needsMeasuring) {
        Write-Note 'Measuring each voice to sort them into male and female (about two minutes).'
        & $venvPython (Join-Path $root 'tools\build_voice_table.py')
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Measurement failed. Run "gv measure" later to finish.' }
    } else {
        Write-Ok 'Voice table already covers the installed voices.'
    }
}

# --- 6. Icon -----------------------------------------------------------------

Write-Step 'Building the icon'
& $venvPython (Join-Path $root 'tools\make_icon.py') | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Warning 'Icon build failed; a drawn fallback icon will be used.' }
else { Write-Ok 'assets\icon.ico' }

# --- 7. Check it works -------------------------------------------------------

Write-Step 'Checking the install'
$report = & $venvPython -c @"
import sys
sys.path.insert(0, r'$root')
problems = []
try:
    from gamevoice.ocr import available_languages
    langs = available_languages()
    print('OCR languages: ' + (', '.join(langs) if langs else 'NONE'))
    if not langs:
        problems.append('Windows has no OCR language pack installed.')
except Exception as exc:
    problems.append(f'Text recognition unavailable: {exc}')

try:
    from gamevoice.voices import VoiceCatalog
    catalog = VoiceCatalog()
    print(f'Voices: {len(catalog.pool(\"male\"))} male, {len(catalog.pool(\"female\"))} female')
except Exception as exc:
    problems.append(f'Voice catalogue unavailable: {exc}')

for problem in problems:
    print('PROBLEM: ' + problem)
"@
$report | ForEach-Object {
    if ($_ -like 'PROBLEM:*') { Write-Warning $_.Substring(9) } else { Write-Ok $_ }
}

# --- 8. Shortcuts ------------------------------------------------------------

if (-not $NoShortcuts) {
    Write-Step 'Creating shortcuts'
    & powershell -ExecutionPolicy Bypass -File (Join-Path $root 'tools\create-shortcut.ps1') | ForEach-Object { Write-Ok $_ }
}

# --- Done --------------------------------------------------------------------

Write-Host ''
Write-Host '  GameVoice is installed.' -ForegroundColor Green
Write-Host ''
Write-Host '  Start it      : GameVoice.cmd, or the Desktop shortcut'
Write-Host '  Read a game   : start the game in BORDERLESS WINDOWED, then press Ctrl+Alt+V'
Write-Host '  If it is quiet: gv doctor --delay 5   (shows exactly what it can see)'
Write-Host ''
Write-Host '  Exclusive fullscreen cannot be captured. Borderless windowed can.' -ForegroundColor Yellow
Write-Host ''
