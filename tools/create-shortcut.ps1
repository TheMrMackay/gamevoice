<#
.SYNOPSIS
    Create a GameVoice shortcut with the proper icon.

.DESCRIPTION
    A .cmd file cannot carry an icon, and launching pythonw.exe directly makes
    Windows show Python's icon. A shortcut fixes both: it points at the app's
    pythonw.exe, sets the icon explicitly, and matches the AppUserModelID the
    app claims at startup so a pinned taskbar button stays ours.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\create-shortcut.ps1
    powershell -ExecutionPolicy Bypass -File tools\create-shortcut.ps1 -StartMenu
#>
[CmdletBinding()]
param(
    [switch]$StartMenu,
    [switch]$Desktop
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $root '.venv\Scripts\pythonw.exe'
$icon = Join-Path $root 'assets\icon.ico'

if (-not (Test-Path $pythonw)) {
    throw "Could not find $pythonw. Create the virtual environment first."
}
if (-not (Test-Path $icon)) {
    Write-Warning "assets\icon.ico is missing; run: .venv\Scripts\python.exe tools\make_icon.py"
}

# Neither switch given means both, which is what someone running this once wants.
if (-not $StartMenu -and -not $Desktop) {
    $StartMenu = $true
    $Desktop = $true
}

$targets = @()
if ($Desktop)   { $targets += [Environment]::GetFolderPath('Desktop') }
if ($StartMenu) { $targets += Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'Microsoft\Windows\Start Menu\Programs' }

$shell = New-Object -ComObject WScript.Shell
foreach ($folder in $targets) {
    if (-not (Test-Path $folder)) {
        Write-Warning "Skipping missing folder: $folder"
        continue
    }
    $path = Join-Path $folder 'GameVoice.lnk'
    $link = $shell.CreateShortcut($path)
    $link.TargetPath = $pythonw
    $link.Arguments = '-m gamevoice.gui.app'
    $link.WorkingDirectory = $root
    $link.Description = 'Read game dialogue aloud, in character'
    if (Test-Path $icon) { $link.IconLocation = "$icon,0" }
    $link.Save()
    Write-Host "Created $path"
}

Write-Host ''
Write-Host 'Pin it to the taskbar from the Start Menu entry if you want it there.'
