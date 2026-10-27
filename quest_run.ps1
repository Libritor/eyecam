# Arm a headset run on the Quest over USB: start the driver hidden, wait for
# the page, push it to the Quest browser through adb reverse, show the log.
# MuseLog must stream to this PC's address on port 5000; the run starts when
# the headset is worn and the trigger pulled.
#
#   .\quest_run.ps1 -Run vision            # tumbling E + O/U pair (about 15 min)
#   .\quest_run.ps1 -Run eight             # the eight colours by planes (about 25 min)
#   .\quest_run.ps1 -Run officer           # the police officer, 8 x 10 (about 33 min)
#   .\quest_run.ps1 -Run wedge             # the step wedge, 12 levels x 4 rows (about 7 min)
#   .\quest_run.ps1 -Run photo             # one of the gallery photos, sealed, 12 x 12 (about 25 min)
#   .\quest_run.ps1 -Run cphoto -Target targets_img/<name>_12x14.png   # a colour poster from make_colour_photo.py
#   .\quest_run.ps1 -Run exam -Ppd 20         # the eye exam, right eye then left (about 15 min; docs/EXAM.md)
#   .\quest_run.ps1 -Run custom -Extra "--mode visual --choice bw --target pix:NO --grid-w 9 --grid-h 5 --spc 8"
#
# -Session names the folder under runs/ (default: <run><n>, the next free number).
param(
    [ValidateSet("vision", "eight", "officer", "wedge", "photo", "cphoto", "grey", "exam", "custom")] [string] $Run = "vision",
    [string] $Session = "",
    [string] $Target = "",
    [string] $Extra = "",
    [int] $SignalTimeout = 7200,
    [int] $Linger = 900,
    [double] $PatchSide = 0.3,   # grey: square flicker patch, fraction of the board side (0 = old 6x4 rule)
    [double] $Ppd = 20           # exam: the browser window's CSS pixels per degree (sets the arcmin scale)
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
    "wedge"   { $args_ = @("--mode", "visual", "--choice", "bw", "--target", "targets_img/wedge_12x4.png",
                           "--grid-w", "12", "--grid-h", "4", "--spc", "8", "--certainty-max-passes", "2",
                           "--rest-every", "48"); $prefix = "vr_wedge" }
    "photo"   { $args_ = @("--mode", "visual", "--choice", "bw", "--target", "targets_img/photo_today.png",
                           "--grid-w", "12", "--grid-h", "12", "--spc", "8", "--certainty-max-passes", "2",
                           "--rest-every", "48"); $prefix = "vr_photo" }
    "cphoto"  { if ($Target -eq "" -or -not (Test-Path $Target)) { "cphoto needs -Target <png at grid size>"; exit 4 }
                Add-Type -AssemblyName System.Drawing
                $img = [System.Drawing.Image]::FromFile((Resolve-Path $Target)); $gw = $img.Width; $gh = $img.Height; $img.Dispose()
                $args_ = @("--mode", "planes", "--color-target", $Target, "--color-grid-w", "$gw", "--color-grid-h", "$gh",
                           "--plane-spc", "5,5,8", "--plane-passes", "1", "--rest-every", "48"); $prefix = "vr_cphoto"
                "colour poster $Target is ${gw} x ${gh}, about $([math]::Round($gw * $gh * 18 / 60)) min" }
    "grey"    { if ($Target -eq "" -or -not (Test-Path $Target)) { "grey needs -Target <grey png at grid size>"; exit 4 }
                Add-Type -AssemblyName System.Drawing
                $img = [System.Drawing.Image]::FromFile((Resolve-Path $Target)); $gw = $img.Width; $gh = $img.Height; $img.Dispose()
                $args_ = @("--mode", "visual", "--choice", "bw", "--target", $Target, "--grid-w", "$gw", "--grid-h", "$gh",
                           "--spc", "8", "--certainty-max-passes", "2", "--rest-every", "48",
                           "--patch-side", "$PatchSide"); $prefix = "vr_grey"
                "grey picture $Target is ${gw} x ${gh}, about $([math]::Round($gw * $gh * 8 / 60)) min a pass (up to two); " +
                "flicker patch a square $([math]::Round($PatchSide * [math]::Max($gw, $gh), 1)) cells wide" }
    "exam"    { $args_ = @("--mode", "exam", "--exam-eye", "right,left", "--exam-ppd", "$Ppd"); $prefix = "vr_exam"
                "eye exam at $Ppd px/deg: right eye then left, about 15 min; cover the other eye with the patch" }
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
if ($Run -eq "photo" -and -not (Test-Path "targets_img\photo_today.png")) {
    python targets_img\make_photo_gallery.py --draw
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
# with the MuseLog phone also on USB, adb needs the Quest named (2026-10-17)
$quest = adb devices -l | Select-String "model:Quest" | ForEach-Object { ($_ -split "\s+")[0] } | Select-Object -First 1
$adb = @(); if ($quest) { $adb = @("-s", $quest); "quest: $quest" } else { "no Quest on adb; using the only device" }
& adb @adb reverse tcp:8082 tcp:8082 | Out-Null
& adb @adb shell input keyevent KEYCODE_WAKEUP
Start-Sleep 1
# -f 0x10000000 (new task): without it a browser that restored last night's tab shows that tab
# and never loads the session page; with it the page opens in front (2026-10-12)
& adb @adb shell "am start -a android.intent.action.VIEW -d 'http://127.0.0.1:8082/?k=quest' -f 0x10000000 com.oculus.browser" | Select-Object -First 1
Start-Sleep 10
Get-Content $log | Where-Object { $_ -notmatch 'pygame|Hello' } | Select-Object -Last 6
"when it is done: python analysis/vision_readout.py runs/$Session --seal targets_img/vision_today_seal.json   (vision)"
"                 python analysis/planes_figures.py runs/$Session --label $Session                            (colour)"
"                 python analysis/wedge_readout.py runs/$Session                                              (wedge)"
"                 python analysis/photo_readout.py runs/$Session --seal targets_img/photo_today_seal.json     (photo)"
