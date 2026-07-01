Add-Type -AssemblyName System.IO.Compression.FileSystem

$docsPath = "c:\Users\parth\3D Objects\Programming\Projects\My Projetcs\The Watcher\Computer_vision_watcher\Documents"
$outputPath = "c:\Users\parth\3D Objects\Programming\Projects\My Projetcs\The Watcher\Computer_vision_watcher\layer7_extracted.txt"

$docName = "Layer7_Storage.docx"
$path = Join-Path $docsPath $docName
$zip = [System.IO.Compression.ZipFile]::OpenRead($path)
$entry = $zip.Entries | Where-Object { $_.FullName -eq "word/document.xml" }
$stream = $entry.Open()
$reader = New-Object System.IO.StreamReader($stream)
$xml = $reader.ReadToEnd()
$reader.Close()
$stream.Close()
$zip.Dispose()
$text = $xml -replace '<[^>]+>', ' '
$text = $text -replace '\s+', ' '
$text.Trim() | Out-File -FilePath $outputPath -Encoding UTF8
Write-Host "Done"
