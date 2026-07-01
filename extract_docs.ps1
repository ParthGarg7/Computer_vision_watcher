Add-Type -AssemblyName System.IO.Compression.FileSystem
$docsPath = "c:\Users\parth\3D Objects\Programming\Projects\My Projetcs\The Watcher\Computer_vision_watcher\Documents"
$outputPath = "c:\Users\parth\3D Objects\Programming\Projects\My Projetcs\The Watcher\Computer_vision_watcher\docs_extracted.txt"
$docs = Get-ChildItem $docsPath -Filter "*.docx"
$output = ""
foreach ($doc in $docs) {
    $output += "=== " + $doc.Name + " ===" + "`n"
    $zip = [System.IO.Compression.ZipFile]::OpenRead($doc.FullName)
    $entry = $zip.Entries | Where-Object { $_.FullName -eq "word/document.xml" }
    if ($entry) {
        $stream = $entry.Open()
        $reader = New-Object System.IO.StreamReader($stream)
        $xml = $reader.ReadToEnd()
        $reader.Close()
        $stream.Close()
        $text = $xml -replace '<[^>]+>', ' '
        $text = $text -replace '\s+', ' '
        $output += $text.Trim() + "`n"
    }
    $zip.Dispose()
    $output += "---END---`n"
}
$output | Out-File -FilePath $outputPath -Encoding UTF8
Write-Host "Done. Output saved to $outputPath"
