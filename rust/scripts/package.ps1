# Build the Windows release: lazy-take-notes.exe in a zip with its SHA-256.
#
#   rust/scripts/package.ps1  -> rust/dist/lazy-take-notes-x86_64-windows.zip
#
# The zip holds one top folder, like the macOS and Linux tarballs.

$ErrorActionPreference = 'Stop'

$rustDir = Split-Path -Parent $PSScriptRoot
$name = 'lazy-take-notes-x86_64-windows'
$stage = Join-Path $rustDir "dist\$name"
$zip = Join-Path $rustDir "dist\$name.zip"

# whisper.cpp defaults to -march=native. A release built for the builder's CPU
# can crash on an older one, so take the baseline.
$env:GGML_NATIVE = 'OFF'

# The release uses whisper.cpp's Vulkan backend. The build needs the SDK's
# glslc and headers; the installed zip needs a GPU driver with Vulkan.
$sdk = Get-ChildItem 'C:\VulkanSDK' -Directory -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $sdk) {
    throw 'Vulkan SDK not found. Install it: winget install KhronosGroup.VulkanSDK'
}
$env:VULKAN_SDK = $sdk.FullName
$env:PATH = "$($sdk.FullName)\Bin;$env:PATH"

# rustup picks the toolchain from rust/ (rust-toolchain.toml).
Push-Location $rustDir
try {
    cargo build --release --locked --features vulkan
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} finally {
    Pop-Location
}

if (Test-Path $stage) { Remove-Item -Path $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage -Force | Out-Null
Copy-Item (Join-Path $rustDir 'target\release\lazy-take-notes.exe') $stage

@'
lazy-take-notes for Windows
Run lazy-take-notes.exe --help.
ffmpeg is required for file transcription: winget install Gyan.FFmpeg
Speech recognition needs a GPU driver with Vulkan support.
Settings: lazy-take-notes.exe config
'@ | Set-Content -Path (Join-Path $stage 'README.txt') -Encoding utf8

Compress-Archive -Path $stage -DestinationPath $zip -Force

$hash = (Get-FileHash -Path $zip -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $name.zip" | Set-Content -Path "$zip.sha256" -Encoding ascii
Write-Output $zip
