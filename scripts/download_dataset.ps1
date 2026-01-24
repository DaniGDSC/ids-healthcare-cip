#!/usr/bin/env pwsh
# Download CIC-IDS-2018 dataset to data/raw/CSE-CIC-IDS2018/
# Uses AWS open bucket with no credentials required.

$ErrorActionPreference = "Stop"

# Get the script directory and set data directory
$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$DataDir = Join-Path $ProjectRoot "data\raw\CSE-CIC-IDS2018"

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
$OriginalDir = Join-Path $DataDir "original"
$ProcessedDir = Join-Path $DataDir "processed_ml"
New-Item -ItemType Directory -Force -Path $OriginalDir | Out-Null
New-Item -ItemType Directory -Force -Path $ProcessedDir | Out-Null

Write-Host "Downloading CIC-IDS-2018 to $DataDir" -ForegroundColor Cyan
Write-Host "This is large (tens of GB). Ensure you have bandwidth and disk space." -ForegroundColor Yellow
Write-Host ""

# Download original network traffic data
Write-Host "Downloading original network traffic and log data..." -ForegroundColor Green
aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Original Network Traffic and Log data/" $OriginalDir --only-show-errors

# Download processed ML data
Write-Host "Downloading processed traffic data for ML algorithms..." -ForegroundColor Green
aws s3 sync --no-sign-request "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" $ProcessedDir --only-show-errors

Write-Host ""
Write-Host "Download complete." -ForegroundColor Green
