<#
.SYNOPSIS
    BharatAlpha Unattended Long-Only Forward Paper-Trading Session Wrapper (Windows).

.DESCRIPTION
    Safely runs a single NSE forward paper-trading session on Windows (e.g., Lenovo Yoga 7i Gen 10):
      1. Prevents Windows sleep cleanly via reversible SetThreadExecutionState
         (ES_CONTINUOUS | ES_SYSTEM_REQUIRED) and ALWAYS restores normal sleep in `finally`.
      2. Checks NSE trading day & Asia/Kolkata (IST) clock directly via Python `zoneinfo`
         (never hard-codes Melbourne UTC+10/+11 offsets so AEDT/AEST DST changes never shift NSE times).
      3. Loads broker read-only credentials from Windows DPAPI encrypted XML or User env vars
         into process-only environment variables without ever printing or logging secrets.
      4. Runs `scripts/run_live_paper_session.py --preflight-only` first and aborts on non-zero exit.
      5. Runs `scripts/run_live_paper_session.py --autonomous-long-only --lock-file ...`.
      6. Runs `scripts/evaluate_forward_paper_soak.py` post-session to refresh evidence reports.

    HARD SAFETY INVARIANTS:
      - Strictly READ-ONLY market data (`LIVE_TRADING_ENABLED=False`, `READ_ONLY_MODE=True`).
      - Standing long-only permission scope; zero real-money or broker order placement.
#>

[CmdletBinding()]
param(
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
    [string]$LockFile = "",

    [Parameter()]
    [string]$SecretsXmlPath = (Join-Path $env:LOCALAPPDATA "BharatAlpha\secrets.xml"),

    [Parameter()]
    [string]$MlModel = "",

    [Parameter()]
    [Nullable[double]]$MinMlProb = $null,

    [Parameter()]
    [switch]$RequirePreOpenWindow,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$SkipPostEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 1. Reversible Sleep Prevention via Win32 SetThreadExecutionState
# ---------------------------------------------------------------------------
$sleepGuardCode = @"
using System;
using System.Runtime.InteropServices;

public static class BharatAlphaSleepGuard
{
    public const uint ES_CONTINUOUS      = 0x80000000;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001;

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

if (-not ("BharatAlphaSleepGuard" -as [type])) {
    Add-Type -TypeDefinition $sleepGuardCode -Language CSharp
}

if ([string]::IsNullOrWhiteSpace($SoakRoot)) {
    $SoakRoot = Join-Path $RepoRoot "data\forward_soak"
}
if ([string]::IsNullOrWhiteSpace($LockFile)) {
    $LockFile = Join-Path $SoakRoot "bharatalpha_soak.lock"
}
if ([string]::IsNullOrWhiteSpace($TokenMapPath)) {
    $TokenMapPath = Join-Path $RepoRoot "config\token_map.json"
}

$LogsDir = Join-Path $SoakRoot "logs"
New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null

function Write-SoakLog {
    param([string]$Message, [string]$Level = "INFO")
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss zzz")
    $line = "[$ts] [$Level] $Message"
    Write-Output $line
    if ($script:CurrentLogFile) {
        Add-Content -Path $script:CurrentLogFile -Value $line -Encoding UTF8
    }
}

function Import-BharatAlphaSecrets {
    param([string]$XmlPath)
    if (Test-Path -LiteralPath $XmlPath) {
        $secureDict = Import-Clixml -Path $XmlPath
        if ($secureDict -is [hashtable]) {
            foreach ($key in $secureDict.Keys) {
                $val = $secureDict[$key]
                if ($val -is [System.Security.SecureString]) {
                    $bstr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($val)
                    try {
                        $plain = [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
                        [Environment]::SetEnvironmentVariable($key, $plain, "Process")
                    }
                    finally {
                        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
                    }
                }
            }
            Write-SoakLog "Loaded encrypted DPAPI broker credentials into process environment (values redacted)."
        }
    }
    else {
        Write-SoakLog "No DPAPI secrets.xml at '$XmlPath'; relying on existing User/Process environment variables."
    }
}

$sleepAcquired = $false
try {
    # Acquire reversible sleep-prevention flag (system stays awake; display may dim normally)
    $prevState = [BharatAlphaSleepGuard]::SetThreadExecutionState(
        [BharatAlphaSleepGuard]::ES_CONTINUOUS -bor [BharatAlphaSleepGuard]::ES_SYSTEM_REQUIRED
    )
    if ($prevState -ne 0) {
        $sleepAcquired = $true
    }

    # -----------------------------------------------------------------------
    # 2. Check NSE Trading Day & Asia/Kolkata Clock via Python zoneinfo
    # -----------------------------------------------------------------------
    $pyCalendarCheck = @"
import json, sys
from datetime import datetime, time
from pathlib import Path
sys.path.insert(0, str(Path(r'$RepoRoot')))
from trader.config import IST_TZ, is_nse_trading_day

now_ist = datetime.now(IST_TZ)
today = now_ist.date()
trading_day = is_nse_trading_day(today)
in_preopen_window = time(8, 30) <= now_ist.time() <= time(9, 25)
print(json.dumps({
    "ist_now": now_ist.isoformat(),
    "ist_date": today.isoformat(),
    "is_nse_trading_day": trading_day,
    "in_preopen_window": in_preopen_window
}))
"@

    $calJsonRaw = & $PythonExe -c $pyCalendarCheck
    if ($LASTEXITCODE -ne 0) {
        throw "Python Asia/Kolkata calendar check failed (exit code $LASTEXITCODE)."
    }
    $calInfo = $calJsonRaw | ConvertFrom-Json
    $istDate = [string]$calInfo.ist_date
    $script:CurrentLogFile = Join-Path $LogsDir ("soak_{0}.log" -f $istDate)

    Write-SoakLog "BharatAlpha Forward Soak Wrapper started. IST=$($calInfo.ist_now) Date=$istDate Provider=$Provider Profile=$Profile"
    if ($sleepAcquired) {
        Write-SoakLog "Reversible sleep guard enabled (SetThreadExecutionState ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."
    }

    if (-not $calInfo.is_nse_trading_day -and -not $DryRun) {
        Write-SoakLog "IST date $istDate is NOT an NSE trading day (weekend or official NSE holiday). Exiting cleanly."
        exit 0
    }

    if ($RequirePreOpenWindow -and -not $calInfo.in_preopen_window -and -not $DryRun) {
        Write-SoakLog "Current IST time ($($calInfo.ist_now)) is outside the 08:30-09:25 IST launch window (DST-safe dual-trigger guard). Exiting cleanly."
        exit 0
    }

    # -----------------------------------------------------------------------
    # 3. Load Secrets Safely (Zero Echoing)
    # -----------------------------------------------------------------------
    Import-BharatAlphaSecrets -XmlPath $SecretsXmlPath

    $sessionDir = Join-Path $SoakRoot $istDate
    New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

    $runnerScript = Join-Path $RepoRoot "scripts\run_live_paper_session.py"
    $evalScript   = Join-Path $RepoRoot "scripts\evaluate_forward_paper_soak.py"

    # Build common runner arguments (never pass credentials on the CLI)
    $baseArgs = @(
        $runnerScript,
        "--provider", $Provider,
        "--symbols"
    ) + $Symbols + @(
        "--token-map", $TokenMapPath,
        "--profile", $Profile,
        "--capital", ([string]$Capital),
        "--output-dir", $sessionDir,
        "--lock-file", $LockFile
    )

    if (-not [string]::IsNullOrWhiteSpace($MlModel)) {
        $baseArgs += @("--ml-model", $MlModel)
    }
    if ($null -ne $MinMlProb) {
        $baseArgs += @("--min-ml-prob", ([string]$MinMlProb))
    }

    # -----------------------------------------------------------------------
    # 4. Stage A: Mandatory Preflight Check (Abort if Non-Zero)
    # -----------------------------------------------------------------------
    Write-SoakLog "Running Stage A preflight check (--preflight-only)..."
    $preflightArgs = $baseArgs + @("--preflight-only")
    & $PythonExe @preflightArgs 2>&1 | ForEach-Object { Write-SoakLog ([string]$_) "PREFLIGHT" }
    $preflightExit = $LASTEXITCODE
    if ($preflightExit -ne 0) {
        Write-SoakLog "Preflight FAILED with exit code $preflightExit. Aborting session (fail-closed)." "ERROR"
        exit $preflightExit
    }
    Write-SoakLog "Stage A preflight PASSED."

    # -----------------------------------------------------------------------
    # 5. Stage B-E: Dry-Run OR Autonomous Long-Only Forward Paper Session
    # -----------------------------------------------------------------------
    if ($DryRun) {
        Write-SoakLog "Running --dry-run validation..."
        $dryArgs = $baseArgs + @("--dry-run")
        & $PythonExe @dryArgs 2>&1 | ForEach-Object { Write-SoakLog ([string]$_) "DRYRUN" }
        $dryExit = $LASTEXITCODE
        Write-SoakLog "Dry-run finished with exit code $dryExit."
        exit $dryExit
    }

    Write-SoakLog "Launching autonomous long-only forward paper session (--autonomous-long-only)..."
    $liveArgs = $baseArgs + @("--autonomous-long-only")
    & $PythonExe @liveArgs 2>&1 | ForEach-Object { Write-SoakLog ([string]$_) "SESSION" }
    $sessionExit = $LASTEXITCODE
    Write-SoakLog "Live paper session exited with code $sessionExit."

    # -----------------------------------------------------------------------
    # 6. Post-Session Forward Soak Evidence Evaluator
    # -----------------------------------------------------------------------
    if (-not $SkipPostEval) {
        Write-SoakLog "Running post-session forward evidence evaluator..."
        & $PythonExe $evalScript --soak-dir $SoakRoot --output-dir $SoakRoot 2>&1 | ForEach-Object {
            Write-SoakLog ([string]$_) "EVAL"
        }
        $evalExit = $LASTEXITCODE
        Write-SoakLog "Post-session evaluator exited with code $evalExit."
    }

    exit $sessionExit
}
finally {
    if ($sleepAcquired) {
        # ALWAYS restore normal Windows sleep policy on exit or error
        [void][BharatAlphaSleepGuard]::SetThreadExecutionState([BharatAlphaSleepGuard]::ES_CONTINUOUS)
        if ($script:CurrentLogFile) {
            Write-SoakLog "Restored normal Windows sleep state (ES_CONTINUOUS)."
        }
    }
}
