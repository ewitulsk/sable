# Planetary foundation native extension

Upstream Sable base: `6966d2928340de7631abcecf8549904b877df0a8` (Minecraft 1.21.1, 2.0.5). Existing compact-body Java/JNI methods and f32 Rapier backend are unchanged.

`rapier/src/foundation.rs` adds ABI 1, a bounded f64 native scene backend for streamed cubic terrain. It uses real Rapier rigid bodies, compound voxel collision, contacts, CCD, fixed constraints, sleeping and queries. Terrain is an exact greedy cuboid partition of occupied 16³ cells. The initial material ABI is binary full-cube/air; arbitrary Minecraft voxel-shape/material integration is later work. A native scene is independent of any Minecraft plot or celestial mass.

Both precisions use Ryan's Rapier source `38e92f117590862481a53df6fc69a5d893e29186` and Parry 0.26.0. The f64 backend needs four focused internal origin-shift methods. `scripts/Prepare-FoundationRapier.ps1` exports the pinned source into an ignored directory and applies `patches/rapier-region-origin.patch`. It retains upstream Apache-2.0 licensing, source revision and patch/file hashes. The original f32 dependency still comes directly from the pinned git revision without this patch.

Translation changes every native body current/queued pose and world COM, collider world pose, cached solver contact point and padded BVH leaf. It preserves local constraint anchors, collider-relative offsets, velocities, sleep flags, modification queues and warm starts. Solver scratch buffers are discarded. Region handles are registry keys, not externally supplied pointers. Calls serialize through a registry mutex and Java owner-thread confinement; concurrent simulation throughput and automatic merging are not claimed.

Prepare generated Rapier before Cargo:

```powershell
./sable_rapier/scripts/Prepare-FoundationRapier.ps1
```

The Planetary Sable repository's `scripts/Build-FoundationNative.ps1` builds the Windows DLL and writes native identity. Its `scripts/Test-FoundationNative.ps1` owns headless fixtures. No upstream installation is changed by these commands. A package replaces the Windows entry inside the existing Sable nested native library; it does not load a second Sable provider.

Initial caps: 64 live scenes; 4,096 registered section keys (including revision tombstones), 4,096 bodies and 4,096 joints per scene; +/-512m local coordinates. Caller must retire/recreate a scene before exhausting its lifetime key budget. Full production streaming/handle recycling is milestone two. The outer world service owns global residency admission. Native scene metadata is never a planet mass source. Dynamic boxes enable a two-meter speculative-contact prediction window; the unchanged hard-CCD switch alone did not satisfy the fixture penetration bound. This is a local-physics quality setting, not a claim that arbitrary flight speeds are qualified.
