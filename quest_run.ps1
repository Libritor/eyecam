# Arm a headset run on the Quest over USB: start the driver hidden, wait for
# the page, push it to the Quest browser through adb reverse, show the log.
# MuseLog must stream to this PC's address on port 5000; the run starts when
# the headset is worn and the trigger pulled.
#
#   .\quest_run.ps1 -Run vision            # tumbling E + O/U pair (about 15 min)
#   .\quest_run.ps1 -Run eight             # the eight colours by planes (about 25 min)
#   .\quest_run.ps1 -Run officer           # the police officer, 8 x 10 (about 33 min)
#   .\quest_run.ps1 -Run custom -Extra "--mode visual --choice bw --target pix:NO --grid-w 9 --grid-h 5 --spc 8"
#
# -Session names the folder under runs/ (default: <run><n>, the next free number).
param(
    [ValidateSet("vision", "eight", "officer", "custom")] [string] $Run = "vision",
    [string] $Session = "",
    [string] $Extra = "",
    [int] $SignalTimeout = 7200,
    [int] $Linger = 900
)
Set-Location $PSScriptRoot
$common = @("--source", "osc", "--freq", "12", "--calib-blocks", "6", "--calib-on", "8", "--calib-off", "8",
            "--calib-style", "bw", "--calib-size", "1", "--page-token", "quest",
            "--signal-timeout", "$SignalTimeout", "--linger", "$Linger")
switch ($Run) {
    "vision"  { $args_ = @("--mode", "visual", "--choice", "bw", "--target", "targets_img/vision_today.png",
                           "--grid-w", "12", "--grid-h", "7", "--spc", "8", "--certainty-max-passes", "2",
                           "--rest-every", "42"); $prefix = "vr_vision" }
    "eight"   { $args_ = @("--mode", "planes", "--color-target", "eight", "--color-grid-w", "8", "--color-grid-h", "4",
                           "--plane-spc", "6,6,10", "--rest-every", "32"); $prefix = "vr_planes" }
    "officer" { $args_ = @("--mode", "planes", "--color-target", "targets_img/officer_8x10.png", "--color-grid-w", "8",
                           "--color-grid-h", "10", "--plane-spc", "5,5,8", "--plane-passes", "1", "--rest-every", "40");
                $prefix = "vr_officer" }
    "custom"  { $args_ = @(); $prefix = "vr_custom" }
}
if ($Extra -ne "") { $args_ += ($Extra -split " ") }
if ($Session -eq "") {
    $n = 1
    while (Test-Path "runs\$prefix$n") { $n++ }
    $Session = "$prefix$n"
}
if ($Run -eq "vision" -and -not (Test-Path "targets_img\vision_today.png")) {
    python targets_img\make_ou_targets.py --draw
}
$busy = netstat -ano | Select-String ":8082 .*LISTENING"
if ($busy) { "port 8082 still held: $busy"; exit 2 }
$udp = netstat -ano -p udp | Select-String ":5000 "
if ($udp) { "udp 5000 still held (an old recorder?): $udp"; exit 3 }
$log = "runs\${Session}_server.log"; $err = "runs\${Session}_server.err"
New-Item -ItemType Directory -Force runs | Out-Null
$all = @("-u", "xr_session.py", "--session", "runs/$Session") + $common + $args_
"python " + ($all -join " ")
$p = Start-Process -FilePath python -ArgumentList $all -WorkingDirectory $PSScriptRoot -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru
"driver pid $($p.Id), session runs/$Session"
$ok = $false
foreach ($i in 1..40) { try { (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8082/" -TimeoutSec 2) | Out-Null; $ok = $true; break } catch { Start-Sleep -Milliseconds 500 } }
"page served: $ok"
adb reverse tcp:8082 tcp:8082 | Out-Null
adb shell input keyevent KEYCODE_WAKEUP
Start-Sleep 1
adb shell "am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8082/?k=quest' com.oculus.browser" | Select-Object -First 1
Start-Sleep 10
Get-Content $log | Where-Object { $_ -notmatch 'pygame|Hello' } | Select-Object -Last 6
"when it is done: python analysis/vision_readout.py runs/$Session --seal targets_img/vision_today_seal.json   (vision)"
"                 python analysis/planes_figures.py runs/$Session --label $Session                            (colour)"
