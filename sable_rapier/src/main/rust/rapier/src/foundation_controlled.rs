//! Controlled actor API 1. Persistent finite-mass solver participants with explicit input/result ownership.
//! No gameplay activation is implied: callers must publish absolute solver poses, never add a second input step.
use super::*;
use rapier3d_f64::parry::bounding_volume::BoundingVolume;

pub(super) const BODY_ID_BASE: i64 = 1_i64 << 62;
const MAX_PLAYERS: usize = 8;
const MAX_ITEMS: usize = 64;
const MAX_CONTACTS: usize = 64;
const MAX_SPEED: f64 = 320.;
// Local accuracy cost for islands touching an actor; existing error tolerances are unchanged.
const ADDITIONAL_SOLVER_ITERATIONS: usize = 8;
#[derive(Clone)]
struct Input {
    sequence: i64,
    start: i64,
    end: i64,
    velocity: Vec3,
    started: bool,
    initial_velocity: Vec3,
}
#[derive(Clone)]
struct Actor {
    id: i64,
    kind: i64,
    lease: i64,
    scene: i64,
    body: i64,
    mass: f64,
    last_sequence: i64,
    input: Option<Input>,
    result: Option<Vec<i64>>,
}
#[derive(Default)]
pub(super) struct State {
    next_lease: i64,
    actors: HashMap<i64, Actor>,
}
fn bits(v: f64) -> i64 {
    v.to_bits() as i64
}
fn vector_bits(v: Vec3) -> [i64; 3] {
    [bits(v.x), bits(v.y), bits(v.z)]
}
pub(super) fn count(state: &State) -> usize {
    state.actors.len()
}
pub(super) fn contains(state: &State, id: i64) -> bool {
    state.actors.contains_key(&id)
}
/// Classification is registry-owned, never inferred from numeric persistent IDs.
pub(super) fn owns_body(state: &State, scene: i64, id: i64) -> bool {
    state
        .actors
        .values()
        .any(|a| a.scene == scene && a.body == id)
}
pub(super) fn owns_scene(state: &State, scene: i64) -> bool {
    state.actors.values().any(|a| a.scene == scene)
}
pub(super) fn retire_scene(state: &mut State, scene: i64) {
    state.actors.retain(|_, a| a.scene != scene);
}
pub(super) fn transfer_ready(
    registry: &Registry,
    source: i64,
    destination: i64,
) -> Result<(), String> {
    if registry.controlled.actors.values().any(|a| {
        (a.scene == source || a.scene == destination) && (a.input.is_some() || a.result.is_some())
    }) {
        return Err("controlled actor input/result retains transfer ownership".into());
    }
    Ok(())
}
/// Called inside the same registry lock immediately after the complete native transfer publishes both scenes.
pub(super) fn transferred(registry: &mut Registry, source: i64, destination: i64) {
    for a in registry
        .controlled
        .actors
        .values_mut()
        .filter(|a| a.scene == source)
    {
        if registry.scenes[&destination].bodies.contains_key(&a.body) {
            a.scene = destination;
        }
    }
}
fn identity(region: &Region, a: &Actor) -> Vec<i64> {
    let h = region.bodies[&a.body];
    let (slot, generation) = h.into_raw_parts();
    vec![
        a.id,
        a.kind,
        a.lease,
        a.scene,
        region.epoch,
        a.body,
        region.body_epochs[&a.body],
        slot as i64,
        generation as i64,
        a.last_sequence,
        region.time_nanos,
    ]
}
fn actor<'a>(registry: &'a Registry, scene: i64, ids: &[i64]) -> Result<&'a Actor, String> {
    if ids.len() < 7 {
        return Err("complete controlled actor registration/body ownership lease required".into());
    }
    let region = transfer::lookup(registry, scene)?;
    let a = registry
        .controlled
        .actors
        .get(&ids[0])
        .ok_or("unknown controlled actor")?;
    if a.kind != ids[1]
        || a.lease != ids[2]
        || a.scene != scene
        || ids[3] != scene
        || ids[4] != region.epoch
        || ids[5] != a.body
        || region.body_epochs.get(&a.body) != Some(&ids[6])
    {
        return Err("stale controlled actor kind/scene/frame/registration/body owner".into());
    }
    Ok(a)
}
fn same_pose(a: &Pose, b: &Pose) -> bool {
    (a.translation - b.translation).length() <= 1e-9
        && (1. - a.rotation.dot(b.rotation).abs()).abs() <= 1e-10
}
/// Validate the actual current scene, including bodies changed since its last solver step.
/// Admission copies no world geometry and fails closed at explicit query limits.
fn validate_shape(
    region: &Region,
    shape: &SharedShape,
    pose: &Pose,
    excluded: Option<RigidBodyHandle>,
) -> Result<(), String> {
    if region.sim.collider_set.len() > 8192 {
        return Err("controlled registration collider query cap".into());
    }
    let bounds = shape.compute_aabb(pose);
    bounded(bounds.mins)?;
    bounded(bounds.maxs)?;
    let mut candidates = 0;
    let mut primitives = 0;
    for (_, collider) in region.sim.collider_set.iter() {
        if !collider.is_enabled() || excluded.is_some() && collider.parent() == excluded {
            continue;
        }
        let current = collider.parent().map_or(*collider.position(), |parent| {
            *region.sim.rigid_body_set[parent].position() * collider.position_wrt_parent().unwrap()
        });
        if !bounds.intersects(&collider.shape().compute_aabb(&current)) {
            continue;
        }
        candidates += 1;
        primitives += collider
            .shape()
            .as_compound()
            .map_or(1, |compound| compound.shapes().len());
        if candidates > 64 || primitives > 16384 {
            return Err("controlled registration candidate/primitive cap".into());
        }
        if rapier3d_f64::parry::query::contact(pose, shape.as_ref(), &current, collider.shape(), 0.)
            .map_err(|_| "unsupported controlled registration contact shape")?
            .is_some_and(|contact| contact.dist < -0.00001)
        {
            return Err("controlled actor starts in penetration".into());
        }
    }
    Ok(())
}
fn actor_body<'a>(region: &'a Region, a: &Actor) -> Result<&'a RigidBody, String> {
    let h = *region
        .bodies
        .get(&a.body)
        .ok_or("controlled actor lost native body")?;
    region
        .sim
        .rigid_body_set
        .get(h)
        .ok_or_else(|| "controlled actor stale body handle".into())
}
/// Whole-scene validation precedes the first velocity write. Each target covers one fixed interval;
/// subdivisions retain solver velocity instead of reapplying controller input after every contact.
pub(super) fn before_step(registry: &mut Registry, scene: i64, nanos: i64) -> Result<(), String> {
    if registry.transfer.is_some() && owns_scene(&registry.controlled, scene) {
        return Err("controlled actors cannot advance during native transfer staging".into());
    }
    let region = transfer::lookup(registry, scene)?;
    let next = region
        .time_nanos
        .checked_add(nanos)
        .ok_or("controlled clock overflow")?;
    let mut start = Vec::new();
    for a in registry
        .controlled
        .actors
        .values()
        .filter(|a| a.scene == scene)
    {
        if a.result.is_some() {
            return Err("unacknowledged controlled actor result blocks scene advancement".into());
        }
        let input = a
            .input
            .as_ref()
            .ok_or("controlled actor lacks exact input for scene step")?;
        if input.start > region.time_nanos
            || input.end < next
            || !input.started && input.start != region.time_nanos
        {
            return Err("controlled actor input does not cover this native step".into());
        }
        let body = actor_body(region, a)?;
        if !body.is_dynamic()
            || (body.mass() - a.mass).abs() > a.mass * 1e-9
            || body.additional_solver_iterations() != ADDITIONAL_SOLVER_ITERATIONS
            || body.gravity_scale() != 0.
            || body.linear_damping() != 0.
            || body.angular_damping() != 0.
            || body.user_force() != Vec3::ZERO
            || body.user_torque() != Vec3::ZERO
        {
            return Err("controlled actor dynamics changed outside its owner".into());
        }
        if !input.started {
            start.push((a.id, a.body, input.velocity));
        }
    }
    let region = registry.scenes.get_mut(&scene).unwrap();
    for (id, key, velocity) in start {
        let body = &mut region.sim.rigid_body_set[region.bodies[&key]];
        body.set_linvel(velocity, true);
        body.set_angvel(Vec3::ZERO, true);
        let input = registry
            .controlled
            .actors
            .get_mut(&id)
            .unwrap()
            .input
            .as_mut()
            .unwrap();
        input.started = true;
        input.initial_velocity = velocity;
    }
    Ok(())
}
/// Results contain absolute poses and solver velocity plus net finite-mass momentum exchange.
/// Contact entries identify real narrow-phase solver pairs; they are not invented per-point forces.
pub(super) fn after_step(registry: &mut Registry, scene: i64) -> Result<(), String> {
    let region = transfer::lookup(registry, scene)?;
    let mut completed = Vec::new();
    for a in registry
        .controlled
        .actors
        .values()
        .filter(|a| a.scene == scene)
    {
        let input = a
            .input
            .as_ref()
            .ok_or("controlled actor input disappeared during native step")?;
        if input.end != region.time_nanos {
            continue;
        }
        let body = actor_body(region, a)?;
        let pose = body.position();
        let velocity = body.linvel();
        bounded(pose.translation)?;
        bounded(velocity)?;
        if velocity.length() > MAX_SPEED {
            return Err(
                "controlled actor solver velocity escaped its admitted speed envelope".into(),
            );
        }
        let impulse = (velocity - input.initial_velocity) * a.mass;
        if !impulse.is_finite() {
            return Err("invalid controlled actor momentum receipt".into());
        }
        let mut contacts: Vec<(i64, i64, i64, i64, Vec3, Vec3)> = Vec::new();
        for pair in region.sim.narrow_phase.contact_pairs() {
            let first = body.colliders().contains(&pair.collider1);
            let second = body.colliders().contains(&pair.collider2);
            if !first && !second {
                continue;
            }
            let other = if first {
                pair.collider2
            } else {
                pair.collider1
            };
            let parent = region.sim.collider_set[other].parent();
            let other_id = parent
                .and_then(|h| {
                    region
                        .bodies
                        .iter()
                        .find(|(_, p)| **p == h)
                        .map(|(id, _)| *id)
                })
                .unwrap_or(0);
            for manifold in &pair.manifolds {
                if manifold.data.solver_contacts.is_empty() {
                    continue;
                }
                if contacts.len() >= MAX_CONTACTS {
                    return Err("controlled actor solver contact receipt cap".into());
                }
                let point = manifold
                    .data
                    .solver_contacts
                    .iter()
                    .map(|p| p.point)
                    .sum::<Vec3>()
                    / (manifold.data.solver_contacts.len() as f64);
                let normal = if first {
                    -manifold.data.normal
                } else {
                    manifold.data.normal
                };
                let other_actor = registry
                    .controlled
                    .actors
                    .values()
                    .find(|other| other.body == other_id && other.scene == scene);
                contacts.push((
                    other_id,
                    region.body_epochs.get(&other_id).copied().unwrap_or(0),
                    other_actor.map_or(0, |other| other.id),
                    other_actor.map_or(0, |other| other.kind),
                    point,
                    normal,
                ));
            }
        }
        contacts.sort_by_key(|c| (c.0, c.1));
        let mut result = identity(region, a);
        result.extend([input.sequence, input.start, input.end]);
        result.extend(vector_bits(pose.translation));
        result.extend([
            bits(pose.rotation.x),
            bits(pose.rotation.y),
            bits(pose.rotation.z),
            bits(pose.rotation.w),
        ]);
        result.extend(vector_bits(velocity));
        result.extend(vector_bits(impulse));
        result.push(contacts.len() as i64);
        for (id, epoch, other_actor, kind, point, normal) in contacts {
            result.extend([id, epoch, other_actor, kind]);
            result.extend(vector_bits(point));
            result.extend(vector_bits(normal));
        }
        completed.push((a.id, result));
    }
    for (id, result) in completed {
        registry.controlled.actors.get_mut(&id).unwrap().result = Some(result);
    }
    Ok(())
}
pub(super) fn dispatch(
    registry: &mut Registry,
    scene: i64,
    op: i32,
    ids: &[i64],
    values: &[f64],
) -> Result<Vec<i64>, String> {
    if registry.transfer.is_some() && !matches!(op, 40 | 42 | 44) {
        return Err("controlled actor mutation forbidden during transfer staging".into());
    }
    match op {
        40 => {
            if !ids.is_empty() {
                return Err("controlled actor capability takes no identities".into());
            }
            require(values, 0)?;
            Ok(vec![
                1,
                MAX_PLAYERS as i64,
                MAX_ITEMS as i64,
                MAX_CONTACTS as i64,
                BODY_ID_BASE,
                bits(MAX_SPEED),
            ])
        }
        41 => {
            if ids.len() != 2 || ids[0] <= 0 || ids[0] >= BODY_ID_BASE || ![1, 2].contains(&ids[1])
            {
                return Err(
                    "controlled actor needs positive bounded ID and PLAYER/ITEM kind".into(),
                );
            }
            require(values, 14)?;
            if registry.controlled.actors.contains_key(&ids[0])
                || character::contains(&registry.characters, ids[0])
                || character::owns_scene(&registry.characters, scene)
            {
                return Err("actor cannot mix controlled and legacy modes".into());
            }
            if registry.controlled.actors.len() + character::count(&registry.characters) >= 128 {
                return Err("shared native actor registration cap".into());
            }
            transfer_ready(registry, scene, scene)?;
            if registry
                .controlled
                .actors
                .values()
                .filter(|a| a.kind == ids[1])
                .count()
                >= if ids[1] == 1 { MAX_PLAYERS } else { MAX_ITEMS }
            {
                return Err("controlled actor kind capacity".into());
            }
            let half = vec(values, 0)?;
            let mass = values[3];
            let pose = transfer::pose(&values[4..])?;
            let velocity = vec(values, 11)?;
            if velocity.length() > MAX_SPEED {
                return Err("controlled initial velocity bound".into());
            }
            if half.min_element() < 0.05
                || half.max_element() > 2.
                || !mass.is_finite()
                || mass < 0.001
                || mass > 500.
            {
                return Err("controlled actor shape/mass bound".into());
            }
            let shape = SharedShape::cuboid(half.x, half.y, half.z);
            validate_shape(transfer::lookup(registry, scene)?, &shape, &pose, None)?;
            let lease = registry
                .controlled
                .next_lease
                .checked_add(1)
                .ok_or("controlled registration exhausted")?;
            let key = BODY_ID_BASE + ids[0];
            let region = registry
                .scenes
                .get_mut(&scene)
                .ok_or("stale controlled scene")?;
            if region.failed_range
                || region.bodies.len() >= 4096
                || region.bodies.contains_key(&key)
            {
                return Err("controlled native body admission refused".into());
            }
            let h = region.sim.rigid_body_set.insert(
                RigidBodyBuilder::dynamic()
                    .pose(pose)
                    .linvel(velocity)
                    .additional_solver_iterations(ADDITIONAL_SOLVER_ITERATIONS)
                    .gravity_scale(0.)
                    .linear_damping(0.)
                    .angular_damping(0.)
                    .lock_rotations()
                    .ccd_enabled(true)
                    .soft_ccd_prediction(2.),
            );
            region.sim.collider_set.insert_with_parent(
                ColliderBuilder::cuboid(half.x, half.y, half.z)
                    .mass(mass)
                    .friction(0.)
                    .friction_combine_rule(CoefficientCombineRule::Min)
                    .restitution(0.)
                    .restitution_combine_rule(CoefficientCombineRule::Min),
                h,
                &mut region.sim.rigid_body_set,
            );
            region.sim.rigid_body_set[h]
                .recompute_mass_properties_from_colliders(&region.sim.collider_set);
            region.bodies.insert(key, h);
            region.body_epochs.insert(key, 0);
            region.mutation += 1;
            let a = Actor {
                id: ids[0],
                kind: ids[1],
                lease,
                scene,
                body: key,
                mass,
                last_sequence: 0,
                input: None,
                result: None,
            };
            let result = identity(region, &a);
            registry.controlled.actors.insert(a.id, a);
            registry.controlled.next_lease = lease;
            Ok(result)
        }
        42 => {
            if ids.len() != 1 {
                return Err("controlled identity requires actor ID".into());
            }
            require(values, 0)?;
            let a = registry
                .controlled
                .actors
                .get(&ids[0])
                .ok_or("unknown controlled actor")?;
            if a.scene != scene {
                return Err("foreign controlled actor".into());
            }
            Ok(identity(transfer::lookup(registry, scene)?, a))
        }
        43 => {
            if ids.len() != 10 {
                return Err(
                    "controlled input requires lease, sequence, exact start/end clock".into(),
                );
            }
            require(values, 10)?;
            let a = actor(registry, scene, ids)?;
            let region = transfer::lookup(registry, scene)?;
            let pose = transfer::pose(values)?;
            let velocity = vec(values, 7)?;
            let duration = ids[9]
                .checked_sub(ids[8])
                .ok_or("controlled input interval overflow")?;
            if a.result.is_some()
                || a.input.is_some()
                || ids[7] <= a.last_sequence
                || ids[8] != region.time_nanos
                || ![12_500_000, 25_000_000, 50_000_000].contains(&duration)
                || velocity.length() > MAX_SPEED
            {
                return Err(
                    "controlled input stale, pending, or outside fixed-step/speed bound".into(),
                );
            }
            if !same_pose(actor_body(region, a)?.position(), &pose) {
                return Err("controlled input start is not the actual native pose".into());
            }
            registry.controlled.actors.get_mut(&ids[0]).unwrap().input = Some(Input {
                sequence: ids[7],
                start: ids[8],
                end: ids[9],
                velocity,
                started: false,
                initial_velocity: velocity,
            });
            Ok(vec![ids[7], ids[8], ids[9]])
        }
        44 => {
            if ids.len() != 7 {
                return Err("controlled result needs exact owner lease".into());
            }
            require(values, 0)?;
            let a = actor(registry, scene, ids)?;
            Ok(a.result.clone().unwrap_or_default())
        }
        45 => {
            if ids.len() != 10 {
                return Err(
                    "controlled acknowledgement needs exact owner/input/clock receipt".into(),
                );
            }
            require(values, 0)?;
            let a = actor(registry, scene, ids)?;
            let result = a.result.as_ref().ok_or("controlled result absent")?;
            if result[11..14] != ids[7..10] {
                return Err("controlled result acknowledgement mismatch".into());
            }
            let a = registry.controlled.actors.get_mut(&ids[0]).unwrap();
            a.last_sequence = ids[7];
            a.result = None;
            a.input = None;
            Ok(vec![a.last_sequence])
        }
        46 => {
            if ids.len() != 7 {
                return Err("controlled retirement needs exact owner lease".into());
            }
            require(values, 0)?;
            let a = actor(registry, scene, ids)?;
            if a.input.is_some() || a.result.is_some() {
                return Err("controlled input/result prevents retirement".into());
            }
            let key = a.body;
            let region = registry.scenes.get_mut(&scene).unwrap();
            let h = region.bodies[&key];
            region
                .sim
                .rigid_body_set
                .remove(
                    h,
                    &mut region.sim.island_manager,
                    &mut region.sim.collider_set,
                    &mut region.sim.impulse_joint_set,
                    &mut region.sim.multibody_joint_set,
                    true,
                )
                .ok_or("controlled body absent")?;
            region.bodies.remove(&key);
            region.body_epochs.remove(&key);
            region.mutation += 1;
            registry.controlled.actors.remove(&ids[0]);
            Ok(vec![])
        }
        47 => {
            if ids.len() != 8 {
                return Err("controlled input abort needs lease and exact sequence".into());
            }
            require(values, 0)?;
            let a = actor(registry, scene, ids)?;
            let input = a.input.as_ref().ok_or("controlled input absent")?;
            if input.started || a.result.is_some() || input.sequence != ids[7] {
                return Err("started controlled interval cannot be aborted".into());
            }
            registry.controlled.actors.get_mut(&ids[0]).unwrap().input = None;
            Ok(vec![])
        }
        48 => {
            if ids.len() != 7 {
                return Err("controlled reconfiguration needs exact owner lease".into());
            }
            require(values, 7)?;
            transfer_ready(registry, scene, scene)?;
            let a = actor(registry, scene, ids)?;
            let region = transfer::lookup(registry, scene)?;
            let half = vec(values, 0)?;
            if half.min_element() < 0.05 || half.max_element() > 2. {
                return Err("controlled shape bound".into());
            }
            let body = actor_body(region, a)?;
            let h = region.bodies[&a.body];
            let center = body.position().translation;
            let pose = transfer::pose(&[
                center.x, center.y, center.z, values[3], values[4], values[5], values[6],
            ])?;
            let shape = SharedShape::cuboid(half.x, half.y, half.z);
            validate_shape(region, &shape, &pose, Some(h))?;
            if body.colliders().len() != 1 {
                return Err("controlled actor collider ownership changed".into());
            }
            let collider = body.colliders()[0];
            let mutation = region
                .mutation
                .checked_add(1)
                .ok_or("controlled mutation exhausted")?;
            let lease = registry
                .controlled
                .next_lease
                .checked_add(1)
                .ok_or("controlled registration exhausted")?;
            let region = registry.scenes.get_mut(&scene).unwrap();
            region.sim.collider_set[collider].set_shape(shape);
            region.sim.rigid_body_set[h].set_position(pose, true);
            region.sim.rigid_body_set[h]
                .recompute_mass_properties_from_colliders(&region.sim.collider_set);
            region.mutation = mutation;
            let a = registry.controlled.actors.get_mut(&ids[0]).unwrap();
            a.lease = lease;
            registry.controlled.next_lease = lease;
            Ok(identity(region, a))
        }
        _ => Err("unknown controlled actor operation".into()),
    }
}
