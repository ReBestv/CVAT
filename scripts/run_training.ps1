param(
    [ValidateSet("unet", "segformer", "deeplab", "evaluate", "preprocess")]
    [string]$Task = "unet",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$PythonExe = "D:\anaconda3\envs\pytorch_env\python.exe"
$ProjectRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Error "Configured Python was not found: $PythonExe"
    exit 1
}

$scriptMap = @{
    unet = "src\train_unet.py"
    segformer = "src\train_segformer.py"
    deeplab = "src\train_deeplab.py"
    evaluate = "src\evaluate.py"
    preprocess = "src\preprocess.py"
}

$TargetScript = Join-Path $ProjectRoot $scriptMap[$Task]
if (-not (Test-Path -LiteralPath $TargetScript)) {
    Write-Error "Target script was not found: $TargetScript"
    exit 1
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "Python: $PythonExe"
Write-Host "Task:   $Task"
Write-Host "Script: $TargetScript"

& $PythonExe $TargetScript @ExtraArgs
exit $LASTEXITCODE
