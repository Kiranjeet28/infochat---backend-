param(
    [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$BaseUrl = $BaseUrl.TrimEnd("/")

function Write-Check([string]$Name, [bool]$Passed, [string]$Expected) {
    $label = if ($Passed) { "PASS" } else { "FAIL" }
    Write-Output ("{0}: {1} (expected {2})" -f $label, $Name, $Expected)
}

function Write-Skipped([string]$Name, [string]$Reason) {
    Write-Output ("SKIPPED: {0} ({1})" -f $Name, $Reason)
}

function Write-HttpErrorStatus($Response) {
    if ($Response.StatusCode -ge 400) {
        Write-Output ("HTTP {0}" -f $Response.StatusCode)
    }
}

function Get-HttpResponseDetails($Response) {
    if ($Response.PSObject.Methods.Name -contains "GetResponseStream") {
        $statusCode = [int]$Response.StatusCode
        $reader = New-Object System.IO.StreamReader($Response.GetResponseStream())
        try { $content = $reader.ReadToEnd() } finally { $reader.Dispose() }
        return @{
            StatusCode = $statusCode
            Content = [string]$content
        }
    }

    $content = $Response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    return @{ StatusCode = [int]$Response.StatusCode; Content = [string]$content }
}

function Invoke-ApiGet([string]$Path) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Method Get -Uri ("{0}{1}" -f $BaseUrl, $Path)
        return @{ StatusCode = [int]$response.StatusCode; Content = [string]$response.Content; Error = $null }
    }
    catch {
        $response = $_.Exception.Response
        if ($null -eq $response) {
            return @{ StatusCode = 0; Content = $null; Error = $_.Exception.Message }
        }
        $details = Get-HttpResponseDetails $response
        return @{ StatusCode = $details.StatusCode; Content = $details.Content; Error = $null }
    }
}

function Invoke-ApiPost([string]$Path, [string]$JsonBody) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Method Post `
            -Uri ("{0}{1}" -f $BaseUrl, $Path) `
            -ContentType "application/json" -Body $JsonBody
        return @{ StatusCode = [int]$response.StatusCode; Content = [string]$response.Content; Error = $null }
    }
    catch {
        $response = $_.Exception.Response
        if ($null -eq $response) {
            return @{ StatusCode = 0; Content = $null; Error = $_.Exception.Message }
        }
        $details = Get-HttpResponseDetails $response
        return @{ StatusCode = $details.StatusCode; Content = $details.Content; Error = $null }
    }
}

$healthResponse = Invoke-ApiGet "/health"
if ($healthResponse.StatusCode -eq 0) {
    Write-Output ("FAIL: GET /health: no response from server: {0}" -f $healthResponse.Error)
    Write-Output ("API not reachable at {0}. Start it with: python -m uvicorn api:app --host 127.0.0.1 --port 8000" -f $BaseUrl)
    return
}
if ($healthResponse.StatusCode -lt 200 -or $healthResponse.StatusCode -ge 300) {
    Write-HttpErrorStatus $healthResponse
    Write-Check "GET /health" $false "HTTP 200 and status=ok"
    Write-Output ("API not reachable at {0}. Start it with: python -m uvicorn api:app --host 127.0.0.1 --port 8000" -f $BaseUrl)
    return
}
$health = $healthResponse.Content | ConvertFrom-Json
if ($health.status -ne "ok") {
    Write-Check "GET /health" $false "status=ok"
    Write-Output ("API not reachable at {0}. Start it with: python -m uvicorn api:app --host 127.0.0.1 --port 8000" -f $BaseUrl)
    return
}
Write-Check "GET /health" $true "status=ok"

$realQuestion = "Which exam centre was combined with another one?"
$realBody = @{ question = $realQuestion } | ConvertTo-Json -Compress
$cacheMissSeconds = $null
$cacheHitSeconds = $null
$cacheMissFound = $false
$cacheHitFound = $false
for ($i = 1; $i -le 4; $i++) {
    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-ApiPost "/api/ask" $realBody
    $watch.Stop()
    $seconds = [math]::Round($watch.Elapsed.TotalSeconds, 3)
    if ($response.StatusCode -eq 0) {
        Write-Output ("FAIL: real-question call {0}: no response from server: {1}" -f $i, $response.Error)
        Write-Check ("real-question call {0} returned a response" -f $i) $false "HTTP response"
        Write-Skipped ("real-question call {0} cached check" -f $i) "no response from server"
        break
    }
    Write-HttpErrorStatus $response
    $payload = $response.Content | ConvertFrom-Json
    $titles = @($payload.sources | ForEach-Object { $_.title } | Where-Object { $_ }) -join "; "
    Write-Output ("Real question call {0}: seconds={1}, status={2}, cached={3}, source_titles={4}" -f `
        $i, $seconds, $payload.status, $payload.cached, $titles)

    if ($payload.status -eq "invalid_citation") {
        Write-Check ("invalid-citation call {0} is not cached" -f $i) ($payload.cached -eq $false) "cached=False"
        Write-Skipped ("real-question call {0} answer" -f $i) "citation validation rejected this answer; retrying"
        continue
    }
    if ($payload.status -ne "answered") {
        Write-Check ("real-question call {0} answered" -f $i) $false "status=answered"
        break
    }

    Write-Check ("real-question call {0} answered" -f $i) $true "status=answered"
    if ($payload.cached -eq $true) {
        if (-not $cacheMissFound) {
            Write-Skipped "first real-question call is uncached" "a valid cached answer already exists; clear this question's cache entry to test a cold miss"
        }
        $cacheHitFound = $true
        $cacheHitSeconds = $seconds
        Write-Check "real-question call is cached" $true "cached=True"
        break
    }

    if (-not $cacheMissFound) {
        $cacheMissFound = $true
        $cacheMissSeconds = $seconds
        Write-Check "first cacheable real-question call is uncached" $true "cached=False"
        continue
    }
    Write-Check "answer was not cached after a valid response" $false "cached=True on the next request"
    break
}
if (-not $cacheMissFound -and -not $cacheHitFound) {
    Write-Check "real question produced a cacheable answer" $false "status=answered"
}
if ($cacheMissFound -and -not $cacheHitFound) {
    Write-Check "cache hit follows a cacheable answer" $false "cached=True on a later request"
}
if ($cacheMissFound -and $cacheHitFound) {
    Write-Check "cached call is faster" ($cacheHitSeconds -lt $cacheMissSeconds) "cached seconds < uncached seconds"
}
else {
    Write-Skipped "cached call is faster" "need an answered cache miss followed by an answered cache hit"
}

$notFoundBody = @{ question = "What is the hostel fee for the next session?" } | ConvertTo-Json -Compress
for ($i = 1; $i -le 2; $i++) {
    $response = Invoke-ApiPost "/api/ask" $notFoundBody
    if ($response.StatusCode -eq 0) {
        Write-Output ("FAIL: not-found call {0}: no response from server: {1}" -f $i, $response.Error)
        Write-Check ("not-found call {0} returned a response" -f $i) $false "HTTP response"
        continue
    }
    Write-HttpErrorStatus $response
    $payload = $response.Content | ConvertFrom-Json
    Write-Output ("Not-found call {0}: status={1}, cached={2}" -f $i, $payload.status, $payload.cached)
    Write-Check ("not-found call {0} is not cached" -f $i) `
        ($payload.status -eq "not_found" -and $payload.cached -eq $false) `
        "status=not_found and cached=False"
}

$emptyBody = @{ question = "" } | ConvertTo-Json -Compress
$emptyResponse = Invoke-ApiPost "/api/ask" $emptyBody
if ($emptyResponse.StatusCode -eq 0) {
    Write-Output ("FAIL: empty-question request: no response from server: {0}" -f $emptyResponse.Error)
    Write-Check "empty question validation" $false "HTTP 400 or 422 with JSON error body"
}
else {
    Write-Output ("Empty-question response: HTTP {0}, body={1}" -f $emptyResponse.StatusCode, $emptyResponse.Content)
    Write-Check "empty question validation" ($emptyResponse.StatusCode -in @(400, 422)) "HTTP 400 or 422 with JSON error body"
}

$longQuestion = "x" * 5000
$longBody = @{ question = $longQuestion } | ConvertTo-Json -Compress
$longResponse = Invoke-ApiPost "/api/ask" $longBody
if ($longResponse.StatusCode -eq 0) {
    Write-Output ("FAIL: 5000-character question request: no response from server: {0}" -f $longResponse.Error)
    Write-Check "long question validation" $false "HTTP 422 with JSON error body"
}
else {
    Write-Output ("5000-character question response: HTTP {0}, body={1}" -f $longResponse.StatusCode, $longResponse.Content)
    Write-Check "long question validation" ($longResponse.StatusCode -eq 422) "HTTP 422 with JSON error body"
}
