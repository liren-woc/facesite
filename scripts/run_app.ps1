param(
    [string]$PythonExe = "D:\anaconda\envs\pytorch\python.exe",
    [string]$GeneratorBackend = "stable_hair",
    [string]$GeneratorRepo = "third_party/Stable-Hair",
    [string]$GeneratorPython = "",
    [string]$StableHairRepo = "third_party/Stable-Hair",
    [string]$StableHairPython = "",
    [string]$OutputDir = "outputs/tryon",
    [string]$Catalog = "data/hairstyles/catalog.example.json",
    [string]$ServerName = "127.0.0.1",
    [int]$ServerPort = 7860
)

[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$stableHairConfigPath = "configs\stable_hair_python.txt"
if ($StableHairPython -eq "" -and $env:STABLE_HAIR_PYTHON) {
    $StableHairPython = $env:STABLE_HAIR_PYTHON
}
if ($StableHairPython -eq "" -and (Test-Path $stableHairConfigPath)) {
    $StableHairPython = (Get-Content $stableHairConfigPath -Raw).Trim().TrimStart([char]0xFEFF)
}
if ($GeneratorPython -eq "") {
    $GeneratorPython = $StableHairPython
}

$env:PYTHONPATH = "src"
$args = @(
    "-m", "hairstyle_tryon.app",
    "--generator-backend", $GeneratorBackend,
    "--generator-repo", $GeneratorRepo,
    "--output-dir", $OutputDir,
    "--catalog", $Catalog,
    "--server-name", $ServerName,
    "--server-port", "$ServerPort"
)

if ($GeneratorPython -ne "") {
    $args += @("--generator-python", $GeneratorPython)
}

& $PythonExe @args
