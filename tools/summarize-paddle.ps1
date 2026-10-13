# Summarizes docs/baseline/paddle/*.json into a markdown table (median ms per stage, img/s, peak WS delta).
param([string]$Dir = "$PSScriptRoot\..\docs\baseline\paddle")
$rows = foreach ($f in Get-ChildItem $Dir -Filter *.json | Sort-Object Name) {
    $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
    $s = $j.summary
    [pscustomobject]@{
        case = $f.BaseName
        total_med = [math]::Round($s.total_ms.median, 1)
        img_s = [math]::Round($s.img_per_s, 2)
        det = [math]::Round($s.stage_ms_mean.det_graph, 1)
        cls = [math]::Round($s.stage_ms_mean.cls_graph, 1)
        rec = [math]::Round($s.stage_ms_mean.rec_graph, 1)
        ws_mb = [math]::Round($s.working_set_mb_delta, 0)
        peak_mb = [math]::Round($j.meta.working_set_mb_peak, 0)
        exact = $j.meta.accuracy.exact_lines
        cer = [math]::Round($j.meta.accuracy.cer * 100, 3)
        clsacc = "$($j.meta.accuracy.cls_correct)/$($j.meta.accuracy.cls_total)"
    }
}
$rows | Format-Table -AutoSize | Out-String -Width 200
