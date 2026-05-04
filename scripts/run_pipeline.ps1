param(
    [Parameter(Mandatory = $true)][string]$Front,
    [string]$Left = "",
    [string]$Right = "",
    [string]$Hairline = "",
    [string]$Crown = "",
    [string]$Shape = "",
    [string]$Color = "",
    [ValidateSet("masculine","feminine","any")][string]$PresentationPreference = "any",
    [ValidateSet("low","medium","high","any")][string]$MaintenancePreference = "any",
    [ValidateSet("auto","cover","balance","open")][string]$ForeheadGoal = "auto",
    [string]$PreferredStyleTag = "any",
    [ValidateSet("teen","young_adult","adult","middle_aged","senior","any")][string]$AgeGroup = "any",
    [string]$SessionLabel = "",
    [string]$SessionNotes = "",
    [switch]$SkipGeneration,
    [string]$PythonExe = "D:\anaconda\envs\pytorch\python.exe",
    [string]$Catalog = "data/hairstyles/catalog.example.json",
    [string]$GeneratorBackend = "stable_hair",
    [string]$GeneratorRepo = "third_party/Stable-Hair",
    [string]$GeneratorPython = "",
    [string]$StableHairRepo = "third_party/Stable-Hair",
    [string]$StableHairPython = "",
    [string]$OutputDir = "outputs/tryon",
    [string]$OutputJson = "outputs/tryon/result.json"
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
    "-m", "hairstyle_tryon.pipeline",
    "--front", $Front,
    "--catalog", $Catalog,
    "--presentation-preference", $PresentationPreference,
    "--maintenance-preference", $MaintenancePreference,
    "--forehead-goal", $ForeheadGoal,
    "--preferred-style-tag", $PreferredStyleTag,
    "--age-group", $AgeGroup,
    "--generator-backend", $GeneratorBackend,
    "--generator-repo", $GeneratorRepo,
    "--output-dir", $OutputDir,
    "--output-json", $OutputJson
)

if ($Left -ne "") { $args += @("--left", $Left) }
if ($Right -ne "") { $args += @("--right", $Right) }
if ($Hairline -ne "") { $args += @("--hairline", $Hairline) }
if ($Crown -ne "") { $args += @("--crown", $Crown) }
if ($Shape -ne "") { $args += @("--shape", $Shape) }
if ($Color -ne "") { $args += @("--color", $Color) }
if ($GeneratorPython -ne "") { $args += @("--generator-python", $GeneratorPython) }
if ($SessionLabel -ne "") { $args += @("--session-label", $SessionLabel) }
if ($SessionNotes -ne "") { $args += @("--session-notes", $SessionNotes) }
if ($SkipGeneration) { $args += "--skip-generation" }

& $PythonExe @args
