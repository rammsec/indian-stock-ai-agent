<#
.SYNOPSIS
    Registers the BharatAlpha Unattended Long-Only Forward Paper-Trading Windows Scheduled Task.

.DESCRIPTION
    Creates a Windows Task Scheduler job (`BharatAlpha-ForwardPaperSoak-LongOnly`) that invokes
    `scripts/windows/run_bharatalpha_soak.ps1` with `-RequirePreOpenWindow`.

    DST-SAFE TIMEZONE DESIGN (Melbourne AEDT/AEST -> NSE Asia/Kolkata):
      - NSE opens at 09:15 IST (Asia/Kolkata, UTC+05:30, no DST).
      - In Melbourne:
          * During AEDT (UTC+11:00, Oct-Apr): 08:50 IST = 14:20 Melbourne local time.
          * During AEST (UTC+10:00, Apr-Oct): 08:50 IST = 13:20 Melbourne local time.
      - To prevent Melbourne DST transitions from EVER shifting NSE launch times by 1 hour,
        this script registers TWO weekday triggers (13:20 and 14:20 local time, customizable).
      - `run_bharatalpha_soak.ps1 -RequirePreOpenWindow` checks `datetime.now(ZoneInfo("Asia/Kolkata"))`
        and `is_nse_trading_day()` via Python:
          * Whichever trigger falls inside 08:30-09:25 IST on an NSE trading day runs the session.
          * The off-season trigger sees it is outside the IST pre-open window and exits cleanly (0).
          * Weekends and official NSE holidays exit cleanly (0).
      - Duplicate-process protection is enforced at TWO layers:
          1. Task Scheduler `MultipleInstances IgnoreNew`
          2. Python `--lock-file` single-instance file lock.

    WAKE-TO-RUN & POWER NOTES (Lenovo Yoga 7i Gen 10):
      - `-WakeToRun` is enabled on the task settings so Windows can wake from sleep if AC power
        and "Allow wake timers" are enabled in Windows Power Options.
      - Keep the laptop plugged into AC power during NSE sessions (08:50-15:40 IST).
      - Credentials are NEVER embedded in Task Scheduler arguments; store them via DPAPI
        (`%LOCALAPPDATA%\BharatAlpha\secrets.xml`) or User environment variables.
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter()]
    [string]$TaskName = "BharatAlpha-ForwardPaperSoak-LongOnly",

    [Parameter()]
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,

    [Parameter()]
    [ValidateSet("zerodha", "angel_one")]
    [string]$Provider = "zerodha",

    [Parameter()]
    [string[]]$Symbols = @("RELIANCE", "TCS", "ITC"),

    [Parameter()]
    [string]$TokenMapPath = "",

    [Parameter()]
    [ValidateSet("BALANCED_ALL", "ORB_BREAKOUT", "SUPERTREND_ADX", "LIQUIDITY_REVERSAL")]
    [string]$Profile = "ORB_BREAKOUT",

    [Parameter()]
    [double]$Capital = 50000.0,

    [Parameter()]
    [string]$SoakRoot = "",

    [Parameter()]
    [string]$PythonExe = "python",

    [Parameter()]
    [string]$AestTriggerLocalTime = "13:20",

    [Parameter()]
    [string]$AedtTriggerLocalTime = "14:20",

    [Parameter()]
    [switch]$Unregister
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Unregister) {
    if ($PSCmdlet.ShouldProcess($TaskName, "Unregister Windows Scheduled Task")) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Output "Unregistered scheduled task '$TaskName' (if it existed)."
    }
    exit 0
}

$wrapperScript = Join-Path $RepoRoot "scripts\windows\run_bharatalpha_soak.ps1"
if (-not (Test-Path -LiteralPath $wrapperScript)) {
    throw "Wrapper script not found: $wrapperScript"
}

if ([string]::IsNullOrWhiteSpace($SoakRoot)) {
    $SoakRoot = Join-Path $RepoRoot "data\forward_soak"
}
if ([string]::IsNullOrWhiteSpace($TokenMapPath)) {
    $TokenMapPath = Join-Path $RepoRoot "config\token_map.json"
}

$symbolsArg = ($Symbols | ForEach-Object { $_.Trim().ToUpper() }) -join ","

# Build PowerShell command arguments (zero secrets in arguments)
$psArgs = @(
    "-NoProfile",
    "-NonInteractive",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$wrapperScript`"",
    "-RepoRoot", "`"$RepoRoot`"",
    "-Provider", $Provider,
    "-Symbols", $symbolsArg,
    "-TokenMapPath", "`"$TokenMapPath`"",
    "-Profile", $Profile,
    "-Capital", ([string]$Capital),
    "-SoakRoot", "`"$SoakRoot`"",
    "-PythonExe", "`"$PythonExe`"",
    "-RequirePreOpenWindow"
) -join " "

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $psArgs `
    -WorkingDirectory $RepoRoot

# Two weekday triggers covering both AEST (UTC+10) and AEDT (UTC+11) so Melbourne DST changes
# never cause drift. The Python Asia/Kolkata guard inside run_bharatalpha_soak.ps1 ensures
# ONLY the trigger landing in 08:30-09:25 IST on an NSE trading day launches a session.
$weekdays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
$triggerAest = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $AestTriggerLocalTime
$triggerAedt = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $weekdays -At $AedtTriggerLocalTime

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -WakeToRun `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

$principal = New-ScheduledTaskPrincipal `
    -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive `
    -RunLevel Limited

$description = (
    "BharatAlpha unattended READ-ONLY long-only forward paper-trading soak runner " +
    "(LIVE_TRADING_ENABLED=False, READ_ONLY_MODE=True). Uses Python Asia/Kolkata timezone " +
    "and NSE holiday guard for DST-safe execution."
)

if ($PSCmdlet.ShouldProcess($TaskName, "Register Windows Scheduled Task")) {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger @($triggerAest, $triggerAedt) `
        -Settings $settings `
        -Principal $principal `
        -Description $description `
        -Force | Out-Null

    Write-Output "Registered Windows Scheduled Task '$TaskName'."
    Write-Output "  Triggers (Mon-Fri): $AestTriggerLocalTime and $AedtTriggerLocalTime local time (IST 08:30-09:25 guarded)"
    Write-Output "  Duplicate protection: MultipleInstances=IgnoreNew + Python --lock-file"
    Write-Output "  WakeToRun: Enabled (ensure AC power + Windows 'Allow wake timers' are enabled)"
    Write-Output "  Safety: LIVE_TRADING_ENABLED=False, READ_ONLY_MODE=True, zero secrets in task definition"
}
