# Windows PowerShell 5.1+; one gcloud copy per file, four concurrent files by default.
[CmdletBinding()]
param(
    [ValidateRange(1,16)][int]$Concurrency = 4,
    [string]$Root = 'C:\Users\liste\Desktop\WaymoDATA',
    [ValidateRange(0,4)][int]$MaxRetries = 4,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root)
$components = @('camera_box','camera_calibration','camera_to_lidar_box_association',
    'lidar_box','lidar_calibration','lidar_camera_projection','lidar_camera_synced_box',
    'lidar_pose','projected_lidar_box','stats','vehicle_pose')
$cloud = (Get-Command gcloud.cmd -ErrorAction Stop).Source
$env:CLOUDSDK_PROXY_TYPE = 'http'
$env:CLOUDSDK_PROXY_ADDRESS = '127.0.0.1'
$env:CLOUDSDK_PROXY_PORT = '7890'
$env:CLOUDSDK_CORE_DISABLE_PROMPTS = '1'
$env:CLOUDSDK_STORAGE_PROCESS_COUNT = '1'
$env:CLOUDSDK_STORAGE_THREAD_COUNT = '1'
$env:CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD = '0'
$env:CLOUDSDK_STORAGE_CHECK_HASHES = 'if_fast_else_fail'

function Read-BatchIds([string]$Component) {
    $script = Join-Path $Root "$Component\download_waymo_train_${Component}_batch4_fast.ps1"
    $body = [IO.File]::ReadAllText($script)
    $match = [regex]::Match($body, '(?s)\$items\s*=\s*@\((.*?)\)')
    if (-not $match.Success) { throw "Cannot find literal batch 4 items in $script" }
    $ids = @([regex]::Matches($match.Groups[1].Value, "'([0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+)'") |
        ForEach-Object { $_.Groups[1].Value })
    if ($ids.Count -ne 50 -or @($ids | Select-Object -Unique).Count -ne 50) { throw "Invalid 50-ID list: $script" }
    return $ids
}
$ids = @(Read-BatchIds 'lidar')
$cameraIds = @(Read-BatchIds 'camera_image')
if (@(Compare-Object $ids $cameraIds).Count) { throw 'Camera and lidar batch 4 lists differ' }
$protected = @()
foreach ($component in @('camera_image','lidar')) {
    $files = @(Get-ChildItem -LiteralPath (Join-Path $Root $component) -File -Filter '*.parquet')
    $expectedNames = @($ids | ForEach-Object { "training_${component}_$_.parquet" })
    if ($files.Count -ne 50 -or @(Compare-Object $expectedNames @($files.Name)).Count) {
        throw "Existing $component files do not match the exact batch 4 list"
    }
    foreach ($file in $files) {
        $protected += [pscustomobject]@{Path=$file.FullName;Bytes=$file.Length;Ticks=$file.LastWriteTimeUtc.Ticks}
    }
}
$logRoot = Join-Path $Root '.batch4-download-logs'
[IO.Directory]::CreateDirectory($logRoot) | Out-Null
$lock = [IO.File]::Open((Join-Path $logRoot 'download.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runLog = Join-Path $logRoot $stamp
[IO.Directory]::CreateDirectory($runLog) | Out-Null
$statusPath = Join-Path $logRoot 'status.json'
function Save-Json($Value, [string]$Path) {
    $temporary = $Path + '.tmp'
    [IO.File]::WriteAllText($temporary, (ConvertTo-Json -InputObject $Value -Depth 12), (New-Object Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
$worker = {
    param($Item, $Cloud, $Retries)
    $ErrorActionPreference = 'Stop'
    $PSNativeCommandUseErrorActionPreference = $false
    function Invoke-Cloud([string[]]$CliArgs) {
        # gcloud uses stderr for normal progress. Only its exit code decides success.
        $ErrorActionPreference = 'Continue'
        $output = @(& $Cloud @CliArgs 2>&1)
        $code = $LASTEXITCODE
        $lines = @($output | ForEach-Object { $_.ToString() })
        [IO.File]::AppendAllText($Item.Log, ((@('> gcloud ' + ($CliArgs -join ' ')) + $lines + @("exit=$code")) -join [Environment]::NewLine) + [Environment]::NewLine)
        return [pscustomobject]@{Code=$code;Lines=$lines}
    }
    $lastError = ''
    for ($attempt = 0; $attempt -le $Retries; $attempt++) {
        try {
            if ($Item.Mode -eq 'metadata') {
                $reply = Invoke-Cloud @('storage','ls','--long',"gs://waymo_open_dataset_v_2_0_1/training/$($Item.Component)/*.parquet")
                if ($reply.Code -ne 0) { throw "Remote listing exit $($reply.Code): $($reply.Lines -join ' ')" }
                $sizes = @{}
                foreach ($line in $reply.Lines) {
                    if ($line -match '^\s*(\d+)\s+\S+\s+gs://waymo_open_dataset_v_2_0_1/training/([^/]+)/([0-9_]+)\.parquet\s*$') {
                        $size = [int64]$Matches[1]; $component = $Matches[2]; $id = $Matches[3]
                        if ($component -eq $Item.Component -and $Item.Ids -contains $id) {
                            if ($sizes.ContainsKey($id) -or $size -le 0) { throw "Invalid/duplicate remote metadata: $id" }
                            $sizes[$id] = $size
                        }
                    }
                }
                if ($sizes.Count -ne 50) { throw "Remote listing contains $($sizes.Count)/50 batch IDs" }
                return [pscustomobject]@{Ok=$true;Component=$Item.Component;Sizes=$sizes;Attempts=($attempt+1)}
            }
            $existing = Get-Item -LiteralPath $Item.Destination -ErrorAction SilentlyContinue
            if ($attempt -eq 0 -and $existing -and -not $existing.PSIsContainer -and $existing.Length -eq $Item.Bytes) {
                return [pscustomobject]@{Ok=$true;Component=$Item.Component;Id=$Item.Id;Bytes=$Item.Bytes;State='existing';Attempts=$attempt}
            }
            # The exact expected destination is the only file cp may replace. Native
            # gcloud trackers and partial downloads are preserved for resumability.
            $reply = Invoke-Cloud @('storage','cp',$Item.Source,$Item.Destination)
            $actual = Get-Item -LiteralPath $Item.Destination -ErrorAction SilentlyContinue
            if ($reply.Code -ne 0) { throw "Copy exit $($reply.Code): $($reply.Lines -join ' ')" }
            if (-not $actual -or $actual.Length -ne $Item.Bytes) { throw "Size mismatch: expected $($Item.Bytes)" }
            return [pscustomobject]@{Ok=$true;Component=$Item.Component;Id=$Item.Id;Bytes=$actual.Length;State='downloaded';Attempts=($attempt+1)}
        } catch {
            $lastError = $_.Exception.Message
            [IO.File]::AppendAllText($Item.Log, "Attempt $($attempt+1) failed: $lastError`r`n")
            if ($attempt -lt $Retries) { Start-Sleep -Seconds ([Math]::Min(30, [Math]::Pow(2,$attempt)*3)) }
        }
    }
    return [pscustomobject]@{Ok=$false;Component=$Item.Component;Id=$Item.Id;Error=$lastError;Log=$Item.Log;Attempts=($Retries+1)}
}
$pool = [RunspaceFactory]::CreateRunspacePool(1,$Concurrency)
$pool.Open()
$script:status = [ordered]@{Started=(Get-Date).ToString('o');Phase='initial_metadata';Total=550;Completed=0;Failures=0;RunLog=$runLog}
function Invoke-Items($Items, [string]$Phase) {
    $pending = New-Object 'Collections.Generic.Queue[object]'
    foreach ($item in $Items) { $pending.Enqueue($item) }
    $active = New-Object 'Collections.Generic.List[object]'
    $results = New-Object 'Collections.Generic.List[object]'
    $script:status.Phase = $Phase; $script:status.Completed = 0; $script:status.Failures = 0
    Save-Json $script:status $statusPath
    try {
        while ($pending.Count -gt 0 -or $active.Count -gt 0) {
            while ($pending.Count -gt 0 -and $active.Count -lt $Concurrency) {
                $item = $pending.Dequeue()
                $ps = [PowerShell]::Create(); $ps.RunspacePool = $pool
                [void]$ps.AddScript($worker.ToString()).AddArgument($item).AddArgument($cloud).AddArgument($MaxRetries)
                $active.Add([pscustomobject]@{PS=$ps;Handle=$ps.BeginInvoke();Item=$item})
            }
            foreach ($job in @($active.ToArray())) {
                if (-not $job.Handle.IsCompleted) { continue }
                try {
                    $values = @($job.PS.EndInvoke($job.Handle))
                    if ($values.Count -ne 1 -or $null -eq $values[0].Ok) { throw "Worker returned $($values.Count) invalid results; $($job.PS.Streams.Error)" }
                    $result = $values[0]
                } catch {
                    $result = [pscustomobject]@{Ok=$false;Component=$job.Item.Component;Id=$job.Item.Id;Error=$_.Exception.Message;Log=$job.Item.Log}
                } finally { $job.PS.Dispose() }
                [void]$active.Remove($job); $results.Add($result)
                $script:status.Completed = $results.Count
                $script:status.Failures = @($results | Where-Object { -not $_.Ok }).Count
                Save-Json $script:status $statusPath
                Write-Host ("{0} {1}/{2} {3} {4} {5}" -f $Phase,$results.Count,$Items.Count,$result.Ok,$result.Component,$result.Id)
            }
            if ($active.Count) { Start-Sleep -Milliseconds 150 }
        }
    } finally {
        foreach ($job in $active) { $job.PS.Stop(); $job.PS.Dispose() }
    }
    return $results.ToArray()
}
function Metadata-Items([string]$Phase) {
    foreach ($component in $components) {
        [pscustomobject]@{Mode='metadata';Component=$component;Ids=$ids;Log=(Join-Path $runLog "${Phase}_${component}.log")}
    }
}
try {
    $metadata = @(Invoke-Items @(Metadata-Items 'initial') 'initial_metadata')
    $metadataFailures = @($metadata | Where-Object { -not $_.Ok })
    if ($metadataFailures.Count) {
        Save-Json @($metadataFailures | ForEach-Object {
            $failure = $_
            foreach ($id in $ids) { [pscustomobject]@{Component=$failure.Component;Id=$id;Error=$failure.Error;Log=$failure.Log} }
        }) (Join-Path $runLog 'metadata-failures.json')
        throw 'Initial remote metadata incomplete; see metadata-failures.json; no downloads started'
    }
    $items = @()
    foreach ($component in $components) {
        $meta = $metadata | Where-Object { $_.Component -eq $component }
        $folder = Join-Path $Root $component
        [IO.Directory]::CreateDirectory($folder) | Out-Null
        foreach ($id in $ids) {
            $items += [pscustomobject]@{Mode='download';Component=$component;Id=$id;Bytes=[int64]$meta.Sizes[$id];
                Source="gs://waymo_open_dataset_v_2_0_1/training/$component/$id.parquet";
                Destination=(Join-Path $folder "training_${component}_$id.parquet");Log=(Join-Path $runLog "${component}_${id}.log")}
        }
    }
    if ($items.Count -ne 550 -or @($items.Destination | Select-Object -Unique).Count -ne 550) { throw 'Invalid 550-file plan' }
    Save-Json $items (Join-Path $runLog 'plan.json')
    if (-not $VerifyOnly) {
        $downloadResults = @(Invoke-Items $items 'download')
        Save-Json $downloadResults (Join-Path $runLog 'download-results.json')
    }
    # A separate fresh remote listing follows all transfers, including skipped files.
    $finalMetadata = @(Invoke-Items @(Metadata-Items 'final') 'final_metadata')
    $checks = @()
    foreach ($item in $items) {
        $meta = $finalMetadata | Where-Object { $_.Component -eq $item.Component }
        $file = Get-Item -LiteralPath $item.Destination -ErrorAction SilentlyContinue
        $remoteBytes = if ($meta.Ok) { [int64]$meta.Sizes[$item.Id] } else { -1 }
        $transfer = if (-not $VerifyOnly) { $downloadResults | Where-Object { $_.Component -eq $item.Component -and $_.Id -eq $item.Id } } else { $null }
        $ok = $meta.Ok -and $null -ne $file -and -not $file.PSIsContainer -and $file.Length -eq $remoteBytes -and $remoteBytes -eq $item.Bytes -and ($VerifyOnly -or $transfer.Ok)
        $checks += [pscustomobject]@{Ok=[bool]$ok;Component=$item.Component;Id=$item.Id;Path=$item.Destination;
            RemoteBytes=$remoteBytes;LocalBytes=$(if($file){$file.Length}else{-1});Error=$(if(-not $meta.Ok){$meta.Error}elseif(-not $VerifyOnly -and -not $transfer.Ok){$transfer.Error}elseif(-not $ok){'Missing file or remote/local size changed'}else{''})}
    }
    foreach ($entry in $protected) {
        $file = Get-Item -LiteralPath $entry.Path
        if ($file.Length -ne $entry.Bytes -or $file.LastWriteTimeUtc.Ticks -ne $entry.Ticks) { throw "Protected input changed: $($entry.Path)" }
    }
    Save-Json $checks (Join-Path $runLog 'verification.json')
    $failures = @($checks | Where-Object { -not $_.Ok })
    Save-Json $failures (Join-Path $runLog 'failures.json')
    $summary = @($checks | Group-Object Component | ForEach-Object {
        [pscustomobject]@{Component=$_.Name;Verified=@($_.Group | Where-Object Ok).Count;Expected=50;Bytes=($_.Group | Measure-Object LocalBytes -Sum).Sum}
    })
    $summary | Format-Table -AutoSize | Out-Host
    Save-Json $summary (Join-Path $runLog 'summary.json')
    $script:status.Phase = $(if($failures.Count){'failed'}else{'completed'})
    $script:status.Completed = 550-$failures.Count; $script:status.Failures = $failures.Count
    $script:status['Finished'] = (Get-Date).ToString('o'); Save-Json $script:status $statusPath
    if ($failures.Count) { throw "$($failures.Count)/550 files failed final verification. Rerun the same script; see $runLog\failures.json" }
    Write-Host 'SUCCESS: 550/550 files independently rechecked against fresh GCS sizes; camera_image/lidar unchanged.'
} catch {
    $script:status.Phase='failed'; $script:status['Error']=$_.Exception.Message
    Save-Json $script:status $statusPath
    throw
} finally {
    $pool.Close(); $pool.Dispose(); $lock.Dispose()
}
