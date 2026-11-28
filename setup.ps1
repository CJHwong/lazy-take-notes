# Install the Windows terminal app from the newest release.
#
#   irm https://raw.githubusercontent.com/CJHwong/lazy-take-notes/main/setup.ps1 | iex
#
# LTN_VERSION=v0.5.0 selects a tag. LTN_RELEASE_URL serves a local folder, for
# the setup smoke test. Nothing is written to the registry; the script prints
# the folder to add to PATH.

$ErrorActionPreference = 'Stop'

$asset = 'lazy-take-notes-x86_64-windows.zip'
$name = $asset -replace '\.zip$', ''
$tag = $env:LTN_VERSION
if (-not $tag) {
    try {
        $latest = Invoke-RestMethod -Uri 'https://api.github.com/repos/CJHwong/lazy-take-notes/releases/latest' `
            -Headers @{ 'User-Agent' = 'lazy-take-notes-setup' }
        $tag = $latest.tag_name
    } catch {
        throw 'Cannot read the newest release. Set $env:LTN_VERSION=vX.Y.Z and run again.'
    }
}
$base = if ($env:LTN_RELEASE_URL) { $env:LTN_RELEASE_URL } else { 'https://github.com/CJHwong/lazy-take-notes/releases/download' }
$dest = Join-Path $env:LOCALAPPDATA "lazy-take-notes\$tag"
$exe = Join-Path $dest "$name\lazy-take-notes.exe"
$shim = Join-Path $env:LOCALAPPDATA 'Programs\lazy-take-notes\bin'

if (Test-Path $exe) {
    Write-Host "lazy-take-notes $tag is already installed"
} else {
    $work = Join-Path $env:TEMP "ltn-$tag-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $work -Force | Out-Null
    try {
        $zip = Join-Path $work $asset
        $shaFile = "$zip.sha256"
        Invoke-WebRequest -Uri "$base/$tag/$asset" -OutFile $zip
        Invoke-WebRequest -Uri "$base/$tag/$asset.sha256" -OutFile $shaFile
        $expected = ((Get-Content $shaFile) -split '\s+')[0].ToLowerInvariant()
        $actual = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $expected) { throw "the download of $asset does not match its SHA-256" }
        New-Item -ItemType Directory -Path $dest -Force | Out-Null
        Expand-Archive -Path $zip -DestinationPath $dest -Force
    } finally {
        Remove-Item -Path $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}

New-Item -ItemType Directory -Path $shim -Force | Out-Null
foreach ($command in @('lazy-take-notes', 'take-note')) {
    $cmd = Join-Path $shim "$command.cmd"
    "@echo off`r`n`"$exe`" %*" | Set-Content -Path $cmd -Encoding ascii
}

Write-Host "Installed lazy-take-notes $tag"
Write-Host "Add this folder to PATH: $shim"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host 'Install ffmpeg for file transcription: winget install Gyan.FFmpeg'
}
