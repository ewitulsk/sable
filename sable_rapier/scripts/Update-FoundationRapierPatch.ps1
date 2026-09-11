param()
$ErrorActionPreference = 'Stop'
$moduleRoot = Split-Path -Parent $PSScriptRoot
$reference = Join-Path $moduleRoot 'build/foundation-rapier-reference'
$generated = Join-Path $moduleRoot 'src/main/rust/planetary-rapier'
$patchPath = Join-Path $moduleRoot 'patches/rapier-region-origin.patch'
$revision = '38e92f117590862481a53df6fc69a5d893e29186'
if ((& git -C $reference rev-parse HEAD).Trim() -ne $revision -or @(& git -C $reference status --porcelain).Count -ne 0) {
    throw 'Patch regeneration requires the unchanged pinned upstream reference'
}
$paths = @([regex]::Matches([IO.File]::ReadAllText($patchPath), '(?m)^diff --git a/(\S+) b/\S+\r?$') | ForEach-Object { $_.Groups[1].Value })
$paths += @('src/dynamics/mod.rs','src/dynamics/solver/solver_body.rs','src/dynamics/solver/velocity_solver.rs',
    'src/dynamics/ccd/ccd_solver.rs','src/dynamics/ccd/toi_entry.rs','src/geometry/broad_phase_bvh.rs','src/geometry/collider.rs','src/geometry/mod.rs','src/pipeline/physics_pipeline.rs','src/pipeline/user_changes.rs')
$lines = [Collections.Generic.List[string]]::new()
$expected = @{}
foreach ($path in ($paths | Sort-Object -Unique)) {
    $before = Join-Path $reference $path
    $after = Join-Path $generated $path
    if (!(Test-Path -LiteralPath $before) -or !(Test-Path -LiteralPath $after)) { throw "Missing bounded source pair: $path" }
    $expected[$path] = [IO.File]::ReadAllText($after).Replace("`r`n", "`n")
    $difference = @(& git diff --no-index --ignore-cr-at-eol --no-ext-diff --no-textconv -- $before $after 2>$null)
    if ($LASTEXITCODE -gt 1) { throw "Cannot compare pinned source: $path" }
    if ($LASTEXITCODE -eq 0) { continue }
    $inHunk = $false
    foreach ($line in $difference) {
        if ($line.StartsWith('@@ ')) { $inHunk = $true }
        if (!$inHunk -and $line.StartsWith('diff --git ')) { $lines.Add("diff --git a/$path b/$path") }
        elseif (!$inHunk -and $line.StartsWith('--- ')) { $lines.Add("--- a/$path") }
        elseif (!$inHunk -and $line.StartsWith('+++ ')) { $lines.Add("+++ b/$path") }
        else { $lines.Add($line.TrimEnd("`r")) }
    }
}
if ($lines.Count -eq 0) { throw 'Refusing to replace the engine patch with an empty diff' }
[IO.File]::WriteAllText($patchPath, (($lines -join "`n") + "`n"), [Text.UTF8Encoding]::new($false))
# Re-extract and apply through the normal source preparation path. This checks that the
# tracked patch alone reproduces the edited generated source, without changing upstream.
& (Join-Path $PSScriptRoot 'Prepare-FoundationRapier.ps1') -ReferencePath $reference
foreach ($path in $expected.Keys) {
    $actual = [IO.File]::ReadAllText((Join-Path $generated $path)).Replace("`r`n", "`n")
    if ($actual -cne $expected[$path]) { throw "Tracked patch did not reproduce generated source: $path" }
}
Write-Output "Pinned Rapier patch reproduced all $($expected.Count) edited source files exactly (normalized line endings)."
