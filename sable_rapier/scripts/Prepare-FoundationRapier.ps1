param([string]$ReferencePath)
$ErrorActionPreference = 'Stop'
$moduleRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $moduleRoot 'src/main/rust'
$generated = Join-Path $sourceRoot 'planetary-rapier'
$patch = Join-Path $moduleRoot 'patches/rapier-region-origin.patch'
$revision = '38e92f117590862481a53df6fc69a5d893e29186'
$patchHash = (Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash
$receiptPath = Join-Path $generated 'planetary-source.json'
if (Test-Path -LiteralPath $receiptPath) {
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.revision -eq $revision -and $receipt.patchSha256 -eq $patchHash) {
        foreach ($entry in $receipt.files) {
            if ((Get-FileHash -LiteralPath (Join-Path $generated $entry.path)).Hash -ne $entry.sha256) { throw "Generated Rapier source changed: $($entry.path)" }
        }
        return
    }
}
if (!$ReferencePath) { $ReferencePath = Join-Path $moduleRoot 'build/foundation-rapier-reference' }
if (!(Test-Path -LiteralPath (Join-Path $ReferencePath '.git'))) {
    & git clone --filter=blob:none --no-checkout 'https://github.com/ryanhcode/rapier.git' $ReferencePath
    if ($LASTEXITCODE -ne 0) { throw 'Cannot fetch pinned Rapier source' }
    & git -C $ReferencePath checkout --detach $revision
    if ($LASTEXITCODE -ne 0) { throw 'Cannot check out pinned Rapier source' }
}
if ((& git -C $ReferencePath rev-parse HEAD) -ne $revision) { throw 'Rapier reference does not match pinned commit' }
New-Item -ItemType Directory -Force -Path $generated | Out-Null
$archive = Join-Path $generated 'source.tar'
& git -C $ReferencePath archive -o $archive $revision Cargo.toml README.md LICENSE src crates/rapier3d-f64
if ($LASTEXITCODE -ne 0) { throw 'Rapier source archive failed' }
& tar -xf $archive -C $generated
if ($LASTEXITCODE -ne 0) { throw 'Rapier source extraction failed' }
$manifestPath = Join-Path $generated 'Cargo.toml'
$manifest = [IO.File]::ReadAllText($manifestPath)
$manifest = [regex]::Replace($manifest,'(?s)members = \[.*?\]','members = ["crates/rapier3d-f64"]',1)
[IO.File]::WriteAllText($manifestPath,$manifest,[Text.UTF8Encoding]::new($false))
& git -C $generated init --quiet
& git -C $generated apply --check --unidiff-zero $patch
if ($LASTEXITCODE -ne 0) { throw 'Pinned origin patch cannot apply cleanly' }
& git -C $generated apply --unidiff-zero $patch
if ($LASTEXITCODE -ne 0) { throw 'Pinned origin patch failed' }
$files = @('Cargo.toml','src/dynamics/rigid_body_set.rs','src/dynamics/rigid_body.rs','src/dynamics/rigid_body_components.rs','src/geometry/collider_set.rs','src/geometry/narrow_phase.rs','src/geometry/broad_phase_bvh.rs') | ForEach-Object { @{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $generated $_)).Hash} }
@{source='https://github.com/ryanhcode/rapier';revision=$revision;patchSha256=$patchHash;license='Apache-2.0';files=$files} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $receiptPath -Encoding utf8
