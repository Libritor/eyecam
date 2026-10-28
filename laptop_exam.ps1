# The eye exam on this laptop's own panel (2560 x 1600 at 240 Hz, 34.45 cm wide),
# with the Muse 2 and its Oz cup read straight over the laptop's Bluetooth
# (muse_ble.py, preset p20 = AUX on; no phone, no Wi-Fi). The page opens in a
# Chrome profile of its own, full screen, on the laptop panel.
#
#   .\laptop_exam.ps1 -Distance 70            # eyes 70 cm from the screen (sets the arcmin scale)
#   .\laptop_exam.ps1 -Distance 70 -Eye right  # one eye only
#   .\laptop_exam.ps1 -Stop                    # end a running session cleanly
#
# Pixels per degree: 2560 px / 34.45 cm = 74.3 px/cm; one degree at D cm is
# D * tan(1 deg) cm, so device px/deg = 74.3 * D * 0.017455 = 1.297 * D. The page
# takes CSS px/deg and multiplies by devicePixelRatio (1.5 at Windows 150 %
# scaling, Chrome zoom 100 %), so --exam-ppd = 1.297 * D / 1.5. At 70 cm one
# arcmin is 1.51 device px: the finest bars asked (2.6, 2.0, 1.3 arcmin) land on
# 4, 3 and 2 device pixels, and the page reports the width it delivered.
# The 240 Hz panel delivers every rate of the sweep exactly: 6, 7.5, 10, 12,
# 15, 20 and 30 Hz are 20, 16, 12, 10, 8, 6 and 4 frames per half period.
param(
    [double] $Distance = 70,
    [string] $Eye = "right,left",
    [string] $Session = "",
    [string] $Extra = "",
    [switch] $Stop
)
Set-Location $PSScriptRoot
$pidFile = Join-Path $env:TEMP "laptop_exam_pids.txt"
if ($Stop) {
    if (Test-Path $pidFile) {
        $parts = (Get-Content $pidFile) -split " "
        $sess = $parts[2]
        if ($sess -and (Test-Path $sess)) { New-Item -ItemType File -Force (Join-Path $sess "stop_recorder") | Out-Null }
        Start-Sleep 3
        foreach ($id in $parts[0..1]) { try { Stop-Process -Id ([int]$id) -Force -ErrorAction Stop; "stopped $id" } catch { "not running $id" } }
        Remove-Item $pidFile
    } else { "nothing recorded as running" }
    exit 0
}
$dpr = 1.5
$ppd = [math]::Round(1.297 * $Distance / $dpr, 2)
if ($Session -eq "") {
    $n = 1
    while (Test-Path "runs\lap_exam$n") { $n++ }
    $Session = "lap_exam$n"
}
$busy = netstat -ano | Select-String ":8082 .*LISTENING"
if ($busy) { "port 8082 still held: $busy"; exit 2 }
$args_ = @("-u", "xr_session.py", "--session", "runs/$Session",
           "--source", "ble", "--freq", "12", "--calib-blocks", "6", "--calib-on", "8", "--calib-off", "8",
           "--calib-style", "bw", "--calib-size", "1", "--page-token", "lap",
           "--signal-timeout", "7200", "--linger", "600",
           "--mode", "exam", "--exam-eye", $Eye, "--exam-ppd", "$ppd",
           "--exam-sizes", "40,28,20,14,10,7,5,3.5,2.6,2,1.3")
if ($Extra -ne "") { $args_ += ($Extra -split " ") }
$log = "runs\${Session}_server.log"; $err = "runs\${Session}_server.err"
New-Item -ItemType Directory -Force runs | Out-Null
"eye exam on the laptop panel at $Distance cm: $ppd CSS px/deg ($([math]::Round($ppd * $dpr, 1)) device px/deg); session runs/$Session"
"python " + ($args_ -join " ")
$p = Start-Process -FilePath python -ArgumentList $args_ -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
$ok = $false
foreach ($i in 1..40) { try { (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8082/" -TimeoutSec 2) | Out-Null; $ok = $true; break } catch { Start-Sleep -Milliseconds 500 } }
"driver pid $($p.Id), page served: $ok"
$prof = Join-Path $env:TEMP "chrome_exam_profile"
$c = Start-Process "C:\Program Files\Google\Chrome\Application\chrome.exe" `
    -ArgumentList @("--user-data-dir=$prof", "--no-first-run", "--no-default-browser-check",
                    "--new-window", "--start-fullscreen", "http://127.0.0.1:8082/?k=lap") -PassThru
"chrome pid $($c.Id) (its own profile, full screen; Esc leaves full screen, close the window when done)"
Set-Content -Path $pidFile -Value "$($p.Id) $($c.Id) runs/$Session"
Start-Sleep 8
Get-Content $log | Where-Object { $_ -notmatch 'pygame|Hello' } | Select-Object -Last 8
"when it is done: python analysis/exam_readout.py runs/$Session   (report: runs/$Session/exam_report.pdf)"
