<#
.SYNOPSIS
    Remove what install.ps1 created.

.DESCRIPTION
    Deletes the virtual environment, the downloaded voice models and the
    shortcuts. Your tuned game profiles and settings are KEPT unless you ask for
    them to go, because they represent work that the download does not.

    Nothing was ever written to the registry, so there is nothing to clean up
    there. The project folder itself is left alone - deleting it is your call,
    and a script cannot sensibly delete the folder it is running from.

.PARAMETER RemoveSettings
    Also delete %LOCALAPPDATA%\GameVoice: your profiles, voice overrides,
    hotkeys and logs. This is the part that cannot be re-downloaded.

.PARAMETER KeepVoices
    Leave the voice models in place. Worth it if you intend to reinstall, since
    they are about 260 MB and take a while to fetch.

.PARAMETER Force
    Do not ask for confirmation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File uninstall.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File uninstall.ps1 -WhatIf

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File uninstall.ps1 -RemoveSettings -Force
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$RemoveSettings,
    [switch]$KeepVoices,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# ConfirmImpact is deliberately left at the default. Marking this High makes
# every ShouldProcess call raise a confirmation prompt, which throws outright in
# a non-interactive shell - so -Force stopped forcing anything. The script asks
# once, in plain language, below.
if ($Force) { $ConfirmPreference = 'None' }

function Write-Step { param([string]$Text) Write-Host "`n==> $Text" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "    $Text" -ForegroundColor Green }
function Write-Note { param([string]$Text) Write-Host "    $Text" -ForegroundColor DarkGray }

function Get-FolderSize {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    try {
        $bytes = (Get-ChildItem -LiteralPath $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
                  Measure-Object -Property Length -Sum).Sum
        return [math]::Round(($bytes / 1MB), 0)
    } catch { return 0 }
}

function Remove-Target {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path $Path)) {
        Write-Note "$Label - not present"
        return $false
    }
    if (-not $PSCmdlet.ShouldProcess($Path, 'Remove')) { return $false }
    try {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        Write-Ok "removed $Label"
        return $true
    } catch {
        Write-Warning "Could not remove $Label`: $($_.Exception.Message)"
        return $false
    }
}

# --- What is actually here ---------------------------------------------------

$userData = if ($env:GAMEVOICE_HOME) { $env:GAMEVOICE_HOME }
            else { Join-Path $env:LOCALAPPDATA 'GameVoice' }

$venv    = Join-Path $root '.venv'
$voices  = Join-Path $root 'voices'
$shortcuts = @(
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'GameVoice.lnk'),
    (Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'Microsoft\Windows\Start Menu\Programs\GameVoice.lnk')
)

Write-Step 'This will remove'
$plan = @()
if (Test-Path $venv)   { $plan += "  the Python environment      .venv\           $(Get-FolderSize $venv) MB" }
if (-not $KeepVoices -and (Test-Path $voices)) {
                         $plan += "  the voice models            voices\          $(Get-FolderSize $voices) MB" }
foreach ($link in $shortcuts) {
    if (Test-Path $link) { $plan += "  a shortcut                  $link" }
}
if ($RemoveSettings -and (Test-Path $userData)) {
    $plan += "  your profiles and settings  $userData"
}

if (-not $plan) {
    Write-Host ''
    Write-Host '  Nothing to remove - GameVoice does not look installed here.' -ForegroundColor Yellow
    Write-Host ''
    exit 0
}
$plan | ForEach-Object { Write-Host $_ }

Write-Step 'This will be kept'
if ($KeepVoices -and (Test-Path $voices)) {
    Write-Note "the voice models ($(Get-FolderSize $voices) MB) - remove with a plain uninstall"
}
if (-not $RemoveSettings) {
    if (Test-Path $userData) {
        Write-Note "your profiles and settings, in $userData"
        Write-Note 'add -RemoveSettings to delete those too'
    }
}
Write-Note "the project folder itself, $root - delete it by hand when you are done"

# --- Confirm -----------------------------------------------------------------

if (-not $Force -and -not $WhatIfPreference) {
    Write-Host ''
    $answer = Read-Host 'Continue? [y/N]'
    if ($answer -notmatch '^(y|yes)$') {
        Write-Host 'Cancelled. Nothing was removed.' -ForegroundColor Yellow
        exit 0
    }
}

# --- Stop it if it is running ------------------------------------------------

Write-Step 'Checking whether GameVoice is running'
$running = @()
try {
    # Matched on the interpreter's own path, not on a command line containing
    # "gamevoice". Only a process running out of THIS folder's .venv is ours -
    # a second copy installed elsewhere must not be killed by this one, and
    # Get-Process avoids the CIM module's noisy auto-import under -WhatIf.
    $running = @(
        Get-Process -Name 'pythonw', 'python' -ErrorAction SilentlyContinue |
            Where-Object {
                $path = $null
                try { $path = $_.Path } catch { }   # access denied on some PIDs
                $path -and $path.StartsWith($venv, [StringComparison]::OrdinalIgnoreCase)
            }
    )
} catch {
    Write-Note 'Could not inspect running processes; continuing.'
}

if ($running) {
    # Files inside .venv stay locked while the interpreter is loaded, so the
    # removal below would half-succeed and leave an unusable directory.
    foreach ($process in $running) {
        if ($PSCmdlet.ShouldProcess("PID $($process.Id)", 'Stop process')) {
            try {
                Stop-Process -Id $process.Id -Force -ErrorAction Stop
                Write-Ok "stopped GameVoice (PID $($process.Id))"
            } catch {
                Write-Warning "Could not stop PID $($process.Id): $($_.Exception.Message)"
            }
        }
    }
    Start-Sleep -Milliseconds 800
} else {
    Write-Note 'not running from this folder'
}

# --- Remove ------------------------------------------------------------------

Write-Step 'Removing'
foreach ($link in $shortcuts) { Remove-Target -Path $link -Label ([IO.Path]::GetFileName($link)) | Out-Null }
Remove-Target -Path $venv -Label '.venv' | Out-Null
if (-not $KeepVoices) { Remove-Target -Path $voices -Label 'voices' | Out-Null }
if ($RemoveSettings)  { Remove-Target -Path $userData -Label 'profiles and settings' | Out-Null }

# Generated at install time but tracked in the repository, so they are restored
# by a checkout rather than lost - left in place deliberately.
Write-Note 'assets\ and data\voice_table.json are part of the repository and were kept'

# --- Done --------------------------------------------------------------------

if ($WhatIfPreference) {
    Write-Host ''
    Write-Host '  Dry run only. Nothing was actually removed.' -ForegroundColor Yellow
    Write-Host ''
    exit 0
}

$leftovers = @($venv, $voices) | Where-Object { Test-Path $_ }
Write-Host ''
if ($leftovers -and -not $KeepVoices) {
    Write-Warning 'Some folders could not be removed:'
    $leftovers | ForEach-Object { Write-Host "    $_" }
    Write-Host '    Close anything using them and run this again.'
} else {
    Write-Host '  GameVoice is uninstalled.' -ForegroundColor Green
}
Write-Host ''
Write-Host "  Reinstall : powershell -ExecutionPolicy Bypass -File install.ps1"
Write-Host "  Finish up : delete $root"
Write-Host ''
