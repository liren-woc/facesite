param(
    [string]$Distro = "Ubuntu",
    [string]$RepoId = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    [string]$LocalDir = "/home/sa/stable-hair-cache/sd15"
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProjectRoot = ($projectRoot -replace "\\", "/")
$wslProjectRoot = "/mnt/" + $wslProjectRoot.Substring(0,1).ToLower() + $wslProjectRoot.Substring(2)
$scriptPath = "$wslProjectRoot/scripts/download_sd15_wsl.sh"
$repoEscaped = $RepoId.Replace("'", "'\''")
$localEscaped = $LocalDir.Replace("'", "'\''")
$command = "$scriptPath '$repoEscaped' '$localEscaped'"

wsl -d $Distro bash -lc $command
if ($LASTEXITCODE -ne 0) {
    throw "Stable Diffusion 1.5 download failed in WSL. Exit code: $LASTEXITCODE"
}

$configPath = "configs\stable_hair_sd15_path.txt"
Set-Content -Path $configPath -Value $LocalDir -Encoding UTF8
Write-Host "Saved Stable-Hair SD1.5 path to $configPath -> $LocalDir"
