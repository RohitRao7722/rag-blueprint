# Generator 4: Microsoft Word "Save as PDF" (wdFormatPDF = 17).
param([string]$Root)
$fonts = @{ en="Calibri"; ru="Arial"; ar="Arial"; ur="Arial"; th="Leelawadee UI"; zh="Microsoft YaHei"; ja="Yu Gothic"; ko="Malgun Gothic" }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
  foreach ($f in Get-ChildItem "$Root\text\*.txt") {
    $code = $f.BaseName
    $font = if ($fonts.ContainsKey($code)) { $fonts[$code] } else { "Nirmala UI" }
    $doc = $word.Documents.Add()
    $txt = [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::UTF8) -replace "`n`n", "`r"
    $doc.Content.Text = $txt
    $r = $doc.Content
    $r.Font.Name = $font; try { $r.Font.NameBi = $font } catch {}
    $r.Font.Size = 12; $r.Font.SizeBi = 12
    if ($code -in @("ar","ur")) { $r.ParagraphFormat.ReadingOrder = 0 }
    $out = "$Root\pdf\${code}__word.pdf"
    $doc.SaveAs2([ref]$out, [ref]17)
    $doc.Close([ref]0)
    Write-Output "$code done"
  }
} finally { $word.Quit() }
