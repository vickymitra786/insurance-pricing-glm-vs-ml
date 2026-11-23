# Convert the Word paper to PDF with Microsoft Word. Author: Vivek Mitra
param([string]$src, [string]$pdf)
$ErrorActionPreference = "Stop"
$src = (Resolve-Path $src).Path

$w = New-Object -ComObject Word.Application
$w.Visible = $false
try {
  $d = $w.Documents.Open($src, $false, $true)
  $d.SaveAs([ref]$pdf, [ref]17)
  $d.Close([ref]0)
} finally { $w.Quit() }
Write-Output "PDF written: $pdf"
