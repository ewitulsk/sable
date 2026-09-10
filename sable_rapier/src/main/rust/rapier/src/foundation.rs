//! ABI 1: opt-in bounded planetary scenes. Ordinary Sable scenes are unchanged.
//! Handles are monotonic registry keys, never caller-supplied native pointers.
use jni::{
    JNIEnv,
    objects::{JClass, JDoubleArray, JIntArray},
    sys::{jdoubleArray, jint, jlong},
};
use rapier3d_f64::prelude::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
#[path = "foundation_transfer.rs"]
mod transfer;
#[path = "foundation_character.rs"]
mod character;

const LIMIT: f64 = 512.0;
const MAX_SECTIONS: usize = 4096;
// Legacy query results carry IDs in doubles. Refuse precision loss until the typed query ABI lands.
const MAX_EXACT_KEY: i64 = (1_i64 << 53) - 1;
const MAX_PARTS_PER_SECTION: usize = 8192;
const MAX_TOTAL_PARTS: usize = 131072;
const MAX_PREPARATIONS: usize = 8;
static LIVE_PARTS: AtomicUsize = AtomicUsize::new(0);
static PREPARATIONS: AtomicUsize = AtomicUsize::new(0);
struct PartBudget(usize);
impl PartBudget {
    fn reserve(parts: usize) -> Result<Self, String> {
        LIVE_PARTS.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |count| count.checked_add(parts).filter(|total| *total <= MAX_TOTAL_PARTS))
            .map_err(|_| "global compound primitive budget exhausted")?;
        Ok(Self(parts))
    }
}
impl Drop for PartBudget { fn drop(&mut self) { LIVE_PARTS.fetch_sub(self.0, Ordering::AcqRel); } }
struct PreparationSlot;
impl PreparationSlot {
    fn reserve() -> Result<Self, String> {
        PREPARATIONS.fetch_update(Ordering::AcqRel, Ordering::Acquire,
            |count| (count < MAX_PREPARATIONS).then_some(count + 1))
            .map_err(|_| "native geometry preparation capacity exhausted")?;
        Ok(Self)
    }
}
impl Drop for PreparationSlot { fn drop(&mut self) { PREPARATIONS.fetch_sub(1, Ordering::AcqRel); } }
struct PreparedShape { shape: Option<SharedShape>, budget: PartBudget, fingerprint: u64, _slot: PreparationSlot }
#[derive(Default)]
struct PreparedRegistry { next: i64, shapes: HashMap<i64, PreparedShape> }
static PREPARED: OnceLock<Mutex<PreparedRegistry>> = OnceLock::new();
struct Section {
    revision: i64,
    body: Option<RigidBodyHandle>,
    collider: Option<ColliderHandle>,
    _parts: PartBudget,
    fingerprint: u64,
    translation: Vec3,
    resident: bool,
}
struct Region {
    sim: Simulation,
    sections: HashMap<i64, Section>,
    bodies: HashMap<i64, RigidBodyHandle>,
    epoch: i64,
    failed_range: bool,
    streamed_terrain: bool,
    section_high_water: i64,
    mutation: i64,
    time_nanos: i64,
    body_epochs: HashMap<i64,i64>,
    joints: HashMap<i64,ImpulseJointHandle>,
    next_legacy_joint: i64,
}
struct Simulation {
    pipeline: PhysicsPipeline,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    island_manager: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    gravity: Vec3,
    parameters: IntegrationParameters,
}
impl Simulation {
    fn validate_bounds(&self, delta: Vec3) -> Result<(), String> {
        for (_, body) in self.rigid_body_set.iter() {
            bounded(body.position().translation - delta)?;
            bounded(body.next_position().translation - delta)?;
        }
        for (_, collider) in self.collider_set.iter() {
            let aabb = collider.compute_aabb();
            bounded(aabb.mins - delta)?;
            bounded(aabb.maxs - delta)?;
            if let Some(parent) = collider.parent() {
                let queued = *self.rigid_body_set[parent].next_position()
                    * collider.position_wrt_parent().unwrap();
                let aabb = collider.shape().compute_aabb(&queued);
                bounded(aabb.mins - delta)?;
                bounded(aabb.maxs - delta)?;
            }
        }
        Ok(())
    }
    fn new(gravity: Vec3) -> Self {
        Self {
            pipeline: PhysicsPipeline::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            gravity,
            parameters: IntegrationParameters {
                max_ccd_substeps: 3,
                normalized_prediction_distance: 0.005,
                normalized_allowed_linear_error: 0.0025,
                normalized_max_corrective_velocity: 50.0,
                ..IntegrationParameters::default()
            },
        }
    }
    fn step(&mut self, dt: f64) {
        self.parameters.dt = dt;
        self.pipeline.step(
            self.gravity,
            &self.parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }
}
#[derive(Default)]
struct Registry {
    next: i64,
    scenes: HashMap<i64, Region>,
    next_transfer: i64,
    transfer: Option<transfer::PreparedTransfer>,
    characters: character::State,
}
static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
fn bounded(v: Vec3) -> Result<Vec3, String> {
    if !v.is_finite() || v.abs().max_element() > LIMIT {
        Err("native coordinate outside +/-512m".into())
    } else {
        Ok(v)
    }
}
fn vec(v: &[f64], at: usize) -> Result<Vec3, String> {
    if v.len() < at + 3
        || v[at..at + 3]
            .iter()
            .any(|x| !x.is_finite() || x.abs() > 512.0)
    {
        return Err("invalid bounded vector".into());
    }
    bounded(Vec3::new(v[at], v[at + 1], v[at + 2]))
}
fn require(v: &[f64], n: usize) -> Result<(), String> {
    if v.len() != n {
        Err(format!("expected {n} values"))
    } else {
        Ok(())
    }
}
fn fingerprint(values: impl Iterator<Item=u64>) -> u64 {
    values.fold(0xcbf29ce484222325, |hash,value| (hash ^ value).wrapping_mul(0x100000001b3))
}

/// Exact greedy cuboid partition of occupied voxels. A section has no internal contact faces
/// between merged cells; unlike a heightfield, arbitrary caves/overhangs remain representable.
fn section_shape(cells: &[i32]) -> (Option<SharedShape>, usize) {
    let mut used = [false; 4096];
    let mut parts = vec![];
    let index = |x: usize, y: usize, z: usize| x + 16 * z + 256 * y;
    for y in 0..16 {
        for z in 0..16 {
            for x in 0..16 {
                if used[index(x, y, z)] || cells[index(x, y, z)] == 0 {
                    continue;
                }
                let mut dx = 1;
                while x + dx < 16 && !used[index(x + dx, y, z)] && cells[index(x + dx, y, z)] != 0 {
                    dx += 1
                }
                let mut dz = 1;
                while z + dz < 16
                    && (x..x + dx)
                        .all(|a| !used[index(a, y, z + dz)] && cells[index(a, y, z + dz)] != 0)
                {
                    dz += 1
                }
                let mut dy = 1;
                while y + dy < 16
                    && (z..z + dz).all(|c| {
                        (x..x + dx)
                            .all(|a| !used[index(a, y + dy, c)] && cells[index(a, y + dy, c)] != 0)
                    })
                {
                    dy += 1
                }
                for b in y..y + dy {
                    for c in z..z + dz {
                        for a in x..x + dx {
                            used[index(a, b, c)] = true;
                        }
                    }
                }
                let half = Vec3::new(dx as f64, dy as f64, dz as f64) * 0.5;
                let center = Vec3::new(x as f64, y as f64, z as f64) + half;
                parts.push((
                    Pose::from_translation(center),
                    SharedShape::cuboid(half.x, half.y, half.z),
                ));
            }
        }
    }
    if parts.is_empty() {
        (None, 0)
    } else {
        let count = parts.len();
        (Some(SharedShape::compound(parts)), count)
    }
}

fn box_shape(values: &[f64], slot: PreparationSlot) -> Result<PreparedShape, String> {
    if values.len() % 6 != 0 || values.len() / 6 > MAX_PARTS_PER_SECTION {
        return Err("invalid compound primitive count".into());
    }
    for p in values.chunks_exact(6) {
        if p.iter().any(|x| !x.is_finite() || *x < 0. || *x > 16.)
            || p[0] >= p[3] || p[1] >= p[4] || p[2] >= p[5] {
            return Err("compound AABBs require finite nonempty cube-local extents in [0,16]".into());
        }
    }
    let budget = PartBudget::reserve(values.len() / 6)?;
    let mut parts = Vec::with_capacity(budget.0);
    for p in values.chunks_exact(6) {
        let min = Vec3::new(p[0],p[1],p[2]);
        let max = Vec3::new(p[3],p[4],p[5]);
        let half = (max-min)*0.5;
        parts.push((Pose::from_translation(min+half),SharedShape::cuboid(half.x,half.y,half.z)));
    }
    let shape = if parts.is_empty() { None } else { Some(SharedShape::compound(parts)) };
    Ok(PreparedShape { shape, budget, fingerprint: fingerprint(values.iter().map(|x| x.to_bits())), _slot: slot })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_planetarysable_world_physics_FoundationNative_invoke<'a>(
    mut env: JNIEnv<'a>,
    _class: JClass<'a>,
    handle: jlong,
    op: jint,
    key: jlong,
    revision: jlong,
    values: JDoubleArray<'a>,
    blocks: JIntArray<'a>,
) -> jdoubleArray {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Vec<f64>, String> {
            let preparation = if op == 1 || op == 14 || op == 20 {
                Some(PreparationSlot::reserve()?)
            } else { None };
            let n = env.get_array_length(&values).map_err(|e| e.to_string())?;
            let b = env.get_array_length(&blocks).map_err(|e| e.to_string())?;
            let max_values = if op == 20 { MAX_PARTS_PER_SECTION * 6 } else { 32 };
            if n as usize > max_values || b > 4096 {
                return Err("oversized foundation payload".into());
            }
            let mut v = vec![0.0; n as usize];
            let mut voxels = vec![0; b as usize];
            env.get_double_array_region(&values, 0, &mut v)
                .map_err(|e| e.to_string())?;
            env.get_int_array_region(&blocks, 0, &mut voxels)
                .map_err(|e| e.to_string())?;
            if op == 20 {
                if b != 0 { return Err("prepared shape does not accept voxel cells".into()); }
                let shape = box_shape(&v, preparation.unwrap())?;
                let mut prepared = PREPARED.get_or_init(|| Mutex::new(PreparedRegistry::default()))
                    .lock().map_err(|_| "prepared geometry registry poisoned")?;
                if prepared.next == MAX_EXACT_KEY { return Err("prepared geometry identity exhausted".into()); }
                prepared.next += 1;
                let id = prepared.next;
                prepared.shapes.insert(id,shape);
                return Ok(vec![id as f64]);
            }
            if op == 21 {
                require(&v,0)?;
                PREPARED.get_or_init(|| Mutex::new(PreparedRegistry::default()))
                    .lock().map_err(|_| "prepared geometry registry poisoned")?
                    .shapes.remove(&key).ok_or("retired prepared geometry")?;
                return Ok(vec![]);
            }
            // Partition legacy masks outside the simulation lock too. Reserve their worst-case
            // primitive count before allocating shape parts, then release the unused reservation.
            let mut binary_shape = if op == 1 || op == 14 {
                if voxels.len() != 4096 || voxels.iter().any(|x| *x != 0 && *x != 1) {
                    return Err("section requires 4096 binary collision cells".into());
                }
                let mut budget = PartBudget::reserve(4096)?;
                let (shape,count) = section_shape(&voxels);
                LIVE_PARTS.fetch_sub(budget.0-count,Ordering::AcqRel); budget.0=count;
                Some(PreparedShape { shape, budget, fingerprint: fingerprint(voxels.iter().map(|x| *x as u64)), _slot: preparation.unwrap() })
            } else { None };
            let mut registry = REGISTRY
                .get_or_init(|| Mutex::new(Registry::default()))
                .lock()
                .map_err(|_| "native registry poisoned")?;
            if op == 0 {
                require(&v, 3)?;
                let gravity = vec(&v, 0)?;
                if registry.scenes.len() >= 64 {
                    return Err("scene cap reached".into());
                }
                registry.next += 1;
                let id = registry.next;
                registry.scenes.insert(
                    id,
                    Region {
                        sim: Simulation::new(gravity),
                        sections: HashMap::new(),
                        bodies: HashMap::new(),
                        epoch: 0,
                        failed_range: false,
                        streamed_terrain: false,
                        section_high_water: 0,
                        mutation: 0,
                        time_nanos: 0,
                        body_epochs: HashMap::new(),
                        joints: HashMap::new(),
                        next_legacy_joint: 0,
                    },
                );
                return Ok(vec![id as f64, 1.0]);
            }
            if op == 10 {
                transfer::retire_scene(&mut registry,handle);
                character::retire_scene(&mut registry.characters,handle);
                registry
                    .scenes
                    .remove(&handle)
                    .ok_or("stale scene handle")?;
                return Ok(vec![]);
            }
            let region = registry
                .scenes
                .get_mut(&handle)
                .ok_or("stale scene handle")?;
            if region.failed_range && op != 5 && op != 13 && op != 18 {
                return Err("scene escaped local bounds; inspect and retire it".into());
            }
            if op == 4 {
                require(&v, 1)?;
                if !v[0].is_finite() || v[0] <= 0. || v[0] > 0.05 {
                    return Err("invalid step".into());
                }
                let elapsed_nanos=(v[0]*1_000_000_000.).round() as i64;
                if elapsed_nanos<=0 {return Err("step is below native clock precision".into());}
                let next_time=region.time_nanos.checked_add(elapsed_nanos).ok_or("simulation clock exhausted")?;
                let next_mutation=region.mutation.checked_add(1).ok_or("scene mutation exhausted")?;
                region.sim.step(v[0]);
                region.time_nanos = next_time;
                region.mutation = next_mutation;
                if let Err(error) = region.sim.validate_bounds(Vec3::ZERO) {
                    region.failed_range = true;
                    return Err(error);
                }
                return Ok(vec![]);
            }
            let result = {
                let sim = &mut region.sim;
                match op {
                    1 | 2 | 14 | 15 | 19 => {
                        let streamed = op == 14 || op == 15 || op == 19;
                        let publishing = op == 1 || op == 14 || op == 19;
                        if streamed && !region.streamed_terrain && !region.sections.is_empty() {
                            return Err("cannot mix legacy and streamed terrain lifetimes".into());
                        }
                        if !streamed && region.streamed_terrain {
                            return Err("streamed terrain requires lease operations".into());
                        }
                        if key <= 0 || revision < 0 {
                            return Err("invalid section key/revision".into());
                        }
                        if streamed && key > MAX_EXACT_KEY {
                            return Err("section lease exceeds exact legacy query identity range".into());
                        }
                        if streamed && !region.sections.contains_key(&key)
                            && (!publishing || key <= region.section_high_water) {
                            return Err("retired or unknown section lease".into());
                        }
                        if region
                            .sections
                            .get(&key)
                            .is_some_and(|s| revision <= s.revision)
                        {
                            return Err("stale section revision".into());
                        }
                        if !region.sections.contains_key(&key)
                            && region.sections.len() >= MAX_SECTIONS
                        {
                            return Err("section key cap reached".into());
                        }
                        let translation = if publishing {
                            require(&v, if op == 19 { 4 } else { 3 })?;
                            let p = vec(&v, 0)?;
                            bounded(p + Vec3::splat(16.))?;
                            p
                        } else {
                            require(&v, 0)?;
                            Vec3::ZERO
                        };
                        // Validate and prepare before removing the live collider. Failed requests leave it intact.
                        let prepared = if op == 19 {
                            if !v[3].is_finite() || v[3] < 1. || v[3] > MAX_EXACT_KEY as f64 || v[3].fract() != 0. {
                                return Err("invalid prepared geometry identity".into());
                            }
                            Some(PREPARED.get_or_init(|| Mutex::new(PreparedRegistry::default()))
                                .lock().map_err(|_| "prepared geometry registry poisoned")?
                                .shapes.remove(&(v[3] as i64)).ok_or("retired prepared geometry")?)
                        } else { binary_shape.take() };
                        let (shape,parts,fingerprint) = match prepared {
                            Some(p) => (p.shape,p.budget,p.fingerprint),
                            None => (None,PartBudget(0),0),
                        };
                        if let Some(old) = region.sections.remove(&key) {
                            if let Some(h) = old.body {
                                sim.rigid_body_set.remove(
                                    h,
                                    &mut sim.island_manager,
                                    &mut sim.collider_set,
                                    &mut sim.impulse_joint_set,
                                    &mut sim.multibody_joint_set,
                                    true,
                                );
                            }
                        }
                        let mut section = Section {
                            revision,
                            body: None,
                            collider: None,
                            _parts: parts,
                            fingerprint,
                            translation,
                            resident: publishing,
                        };
                        if publishing {
                            if let Some(shape) = shape {
                                let body = sim
                                    .rigid_body_set
                                    .insert(RigidBodyBuilder::fixed().translation(translation));
                                let collider = sim.collider_set.insert_with_parent(
                                    ColliderBuilder::new(shape)
                                        .density(0.)
                                        .friction(0.45)
                                        .build(),
                                    body,
                                    &mut sim.rigid_body_set,
                                );
                                section.body = Some(body);
                                section.collider = Some(collider);
                            }
                        }
                        if streamed {
                            region.streamed_terrain = true;
                            region.section_high_water = region.section_high_water.max(key);
                        }
                        if op != 15 { region.sections.insert(key, section); }
                        Ok(vec![revision as f64])
                    }
                    3 | 11 => {
                        require(&v, 6)?;
                        let p = vec(&v, 0)?;
                        let half = vec(&v, 3)?;
                        bounded(p + half)?;
                        bounded(p - half)?;
                        if key <= 0
                            || region.bodies.contains_key(&key)
                            || half.min_element() <= 0.
                            || half.max_element() > 16.
                        {
                            return Err("invalid body key/extents".into());
                        }
                        if region.bodies.len() >= 4096 {
                            return Err("body cap reached".into());
                        }
                        let builder = if op == 3 {
                            RigidBodyBuilder::dynamic()
                                .ccd_enabled(true)
                                .soft_ccd_prediction(2.0)
                        } else {
                            RigidBodyBuilder::kinematic_position_based()
                        };
                        let h = sim.rigid_body_set.insert(builder.translation(p));
                        sim.collider_set.insert_with_parent(
                            ColliderBuilder::cuboid(half.x, half.y, half.z)
                                .mass(1.)
                                .friction(0.45)
                                .build(),
                            h,
                            &mut sim.rigid_body_set,
                        );
                        region.bodies.insert(key, h);
                        region.body_epochs.insert(key,0);
                        Ok(vec![key as f64])
                    }
                    5 => {
                        let h = *region.bodies.get(&key).ok_or("unknown body")?;
                        let rb = &sim.rigid_body_set[h];
                        let p = rb.position();
                        let v = rb.linvel();
                        let w = rb.angvel();
                        let next = rb.next_position();
                        let mut contact_count = 0.;
                        let mut penetration = 0.0f64;
                        for pair in sim.narrow_phase.contact_pairs() {
                            if rb.colliders().contains(&pair.collider1)
                                || rb.colliders().contains(&pair.collider2)
                            {
                                for manifold in &pair.manifolds {
                                    for point in &manifold.points {
                                        if point.dist <= 0. {
                                            contact_count += 1.;
                                            penetration = penetration.max(-point.dist);
                                        }
                                    }
                                }
                            }
                        }
                        Ok(vec![
                            p.translation.x as f64,
                            p.translation.y as f64,
                            p.translation.z as f64,
                            p.rotation.x as f64,
                            p.rotation.y as f64,
                            p.rotation.z as f64,
                            p.rotation.w as f64,
                            v.x as f64,
                            v.y as f64,
                            v.z as f64,
                            w.x as f64,
                            w.y as f64,
                            w.z as f64,
                            if rb.is_sleeping() { 1. } else { 0. },
                            contact_count,
                            penetration as f64,
                            next.translation.x as f64,
                            next.translation.y as f64,
                            next.translation.z as f64,
                            h.into_raw_parts().0 as f64,
                            h.into_raw_parts().1 as f64,
                        ])
                    }
                    6 => {
                        require(&v, 3)?;
                        let delta = vec(&v, 0)?;
                        if revision != region.epoch + 1 {
                            return Err("stale frame epoch".into());
                        }
                        sim.validate_bounds(delta)?;
                        sim.rigid_body_set.planetary_shift_origin(delta);
                        sim.collider_set.planetary_shift_origin(delta);
                        sim.narrow_phase.planetary_shift_origin(delta);
                        sim.broad_phase
                            .planetary_shift_origin(delta, &sim.collider_set);
                        sim.ccd_solver = CCDSolver::new();
                        sim.pipeline = PhysicsPipeline::new();
                        region.epoch = revision;
                        for section in region.sections.values_mut() { section.translation -= delta; }
                        Ok(vec![region.epoch as f64])
                    }
                    7 => {
                        require(&v, 7)?;
                        let origin = vec(&v, 0)?;
                        let direction = vec(&v, 3)?;
                        if direction.length_squared() < 0.99
                            || direction.length_squared() > 1.01
                            || !v[6].is_finite()
                            || v[6] <= 0.
                            || v[6] > 1024.
                        {
                            return Err("invalid ray".into());
                        }
                        let ray = Ray::new(origin, direction.normalize());
                        let mut nearest = v[6];
                        let mut hit = -1.;
                        for (key, s) in &region.sections {
                            if let Some(h) = s.collider {
                                let c = &sim.collider_set[h];
                                if let Some(t) =
                                    c.shape().cast_ray(c.position(), &ray, nearest, true)
                                {
                                    nearest = t;
                                    hit = *key as f64;
                                }
                            }
                        }
                        Ok(vec![hit, nearest as f64])
                    }
                    8 => {
                        require(&v, 6)?;
                        if sim.impulse_joint_set.len() >= 4096 {
                            return Err("joint cap reached".into());
                        }
                        let a = *region.bodies.get(&key).ok_or("unknown joint body A")?;
                        let b = *region.bodies.get(&revision).ok_or("unknown joint body B")?;
                        if a == b {
                            return Err("self joint".into());
                        }
                        let j = FixedJointBuilder::new()
                            .local_anchor1(vec(&v, 0)?)
                            .local_anchor2(vec(&v, 3)?)
                            .contacts_enabled(false);
                        loop {
                            region.next_legacy_joint=region.next_legacy_joint.checked_sub(1).ok_or("legacy joint identity exhausted")?;
                            if !region.joints.contains_key(&region.next_legacy_joint) {break;}
                        }
                        let joint=sim.impulse_joint_set.insert(a, b, j, true);
                        region.joints.insert(region.next_legacy_joint,joint);
                        Ok(vec![])
                    }
                    9 => {
                        let h = *region.bodies.get(&key).ok_or("unknown body")?;
                        sim.rigid_body_set[h].sleep();
                        Ok(vec![])
                    }
                    12 => {
                        require(&v, 3)?;
                        let h = *region.bodies.get(&key).ok_or("unknown body")?;
                        let p = vec(&v, 0)?;
                        for collider in sim.rigid_body_set[h].colliders() {
                            let c = &sim.collider_set[*collider];
                            let movement = p - sim.rigid_body_set[h].position().translation;
                            let aabb = c.compute_aabb();
                            bounded(aabb.mins + movement)?;
                            bounded(aabb.maxs + movement)?;
                        }
                        let rb = &mut sim.rigid_body_set[h];
                        if !rb.is_kinematic() {
                            return Err("body not kinematic".into());
                        }
                        rb.set_next_kinematic_translation(p);
                        Ok(vec![])
                    }
                    13 => Ok(vec![
                        region.epoch as f64,
                        region
                            .sections
                            .values()
                            .filter(|s| s.body.is_some())
                            .count() as f64,
                        region.bodies.len() as f64,
                        sim.impulse_joint_set.len() as f64,
                        if region.failed_range { 1. } else { 0. },
                    ]),
                    16 => { require(&v, 0)?; Ok(vec![2.0]) },
                    17 => {
                        require(&v, 0)?;
                        let body = *region.bodies.get(&key).ok_or("unknown body")?;
                        sim.rigid_body_set.remove(body, &mut sim.island_manager,
                            &mut sim.collider_set, &mut sim.impulse_joint_set,
                            &mut sim.multibody_joint_set, true).ok_or("stale body registry")?;
                        region.bodies.remove(&key);
                        region.body_epochs.remove(&key);
                        region.joints.retain(|_,joint| sim.impulse_joint_set.get(*joint).is_some());
                        Ok(vec![])
                    },
                    18 => {
                        require(&v, 0)?;
                        Ok(vec![region.sections.len() as f64,
                            region.sections.values().filter(|s| s.body.is_some()).count() as f64,
                            region.bodies.len() as f64, sim.impulse_joint_set.len() as f64,
                            region.section_high_water as f64])
                    },
                    22 | 23 | 24 => {
                        require(&v,if op == 23 { 0 } else { 3 })?;
                        let field = if op == 23 { None } else { Some(vec(&v,0)?) };
                        let body = *region.bodies.get(&key).ok_or("unknown body")?;
                        let body = &mut sim.rigid_body_set[body];
                        if !body.is_dynamic() { return Err("gravity/force requires dynamic body".into()); }
                        if op == 24 { body.add_force(field.unwrap(),true); }
                        else { body.planetary_set_gravity(field); }
                        Ok(vec![])
                    },
                    25 => { require(&v,0)?; Ok(vec![LIVE_PARTS.load(Ordering::Acquire) as f64,
                        PREPARATIONS.load(Ordering::Acquire) as f64]) },
                    26 => { require(&v,0)?; transfer::body_state(region,key,revision) },
                    _ => Err("unknown foundation operation".into()),
                }
            };
            if result.is_ok() && !matches!(op,5|7|13|16|18|25|26) { region.mutation += 1; }
            result
        },
    ));
    let output = match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            let _ = env.throw_new("java/lang/IllegalArgumentException", e);
            return std::ptr::null_mut();
        }
        Err(_) => {
            let _ = env.throw_new("java/lang/IllegalStateException", "foundation native panic");
            return std::ptr::null_mut();
        }
    };
    match env.new_double_array(output.len() as i32) {
        Ok(a) => {
            if env.set_double_array_region(&a, 0, &output).is_err() {
                return std::ptr::null_mut();
            }
            a.into_raw()
        }
        Err(_) => std::ptr::null_mut(),
    }
}
