param(
    [string]$Distro = "Ubuntu",
    [string]$EnvName = "stablehair"
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$wslProjectRoot = "/mnt/" + $projectRoot.Substring(0,1).ToLower() + $projectRoot.Substring(2).Replace("\","/")
$createEnvScript = "$wslProjectRoot/scripts/create_stable_hair_env.sh"
$configPath = Join-Path $projectRoot "configs\stable_hair_python.txt"
$escapedEnvName = [System.Management.Automation.Language.CodeGeneration]::EscapeSingleQuotedStringContent($EnvName)

wsl -d $Distro bash -lc "export STABLE_HAIR_ENV_NAME='$escapedEnvName'; $createEnvScript"

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($configPath, "wsl://$Distro", $utf8NoBom)

Write-Host "Stable-Hair WSL environment is ready."
Write-Host "  distro : $Distro"
Write-Host "  env    : $EnvName"
Write-Host "  config : $configPath"
Write-Host ""
Write-Host "Validate with:"
Write-Host "  D:\anaconda\envs\pytorch\python.exe scripts\check_stable_hair_backend.py --stable-hair-python wsl://$Distro"
