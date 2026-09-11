//! Controlled actor API 1. Persistent finite-mass solver participants with explicit input/result ownership.
//! No gameplay activation is implied: callers must publish absolute solver poses, never add a second input step.
use super::*;
use rapier3d_f64::parry::bounding_volume::BoundingVolume;

pub(super) const BODY_ID_BASE: i64 = 1_i64 << 62;
const MAX_PLAYERS: usize = 8;
const MAX_ITEMS: usize = 64;
const MAX_CONTACTS: usize = 64;
const MAX_SPEED: f64 = 320.;
const MAX_MOTOR_SEGMENTS: usize = 42;
#[derive(Clone)]
struct MotorSegment { end: i64, velocity: Vec3 }
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
    segments: Vec<MotorSegment>,
    active_segment: usize,
    applied_motor_delta: Vec3,
    terminal: Option<Vec<i64>>,
    terminal_supported: bool,
    // Actual force events from all CCD subdivisions; never endpoint support contacts.
    events: Vec<[i64; 16]>,
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
#[derive(Clone, Default)]
pub(super) struct State {
    next_lease: i64,
    actors: HashMap<i64, Actor>,
}
/// One bounded collector per actual outer native step, constructed before any physics mutation.
/// Rapier invokes the force handler inside CCD subdivisions. Its API exposes dt, not the
/// subdivision's absolute start, so each event reports the honest enclosing native-step bracket.
#[derive(Clone, Copy, Default)]
struct Peer { body: i64, epoch: i64, actor: i64, kind: i64, registration: i64 }
struct EventBuffer {
    counts: HashMap<i64, usize>,
    rows: Vec<(i64, [i64; 16])>,
    terminal_ineligible: std::collections::HashSet<i64>,
    failed: bool,
}
pub(super) struct StepEvents {
    start: i64,
    end: i64,
    peers: HashMap<RigidBodyHandle, Peer>,
    buffer: Mutex<EventBuffer>,
}
impl StepEvents {
    pub(super) fn prepare(registry: &Registry, scene: i64, end: i64) -> Result<Self, String> {
        let region = transfer::lookup(registry, scene)?;
        if !owns_scene(&registry.controlled, scene) {
            return Ok(Self { start: region.time_nanos, end, peers: HashMap::new(),
                buffer: Mutex::new(EventBuffer { counts: HashMap::new(), rows: Vec::new(), terminal_ineligible: std::collections::HashSet::new(), failed: false }) });
        }
        if region.bodies.len() > 4096 || region.sections.len() > MAX_SECTIONS {
            return Err("interval history body/terrain registry cap".into());
        }
        let mut peers = HashMap::with_capacity(region.bodies.len()+region.sections.len());
        for section in region.sections.values() {
            if let Some(handle)=section.body { peers.insert(handle,Peer::default()); }
        }
        for (id, handle) in &region.bodies {
            peers.insert(*handle, Peer { body: *id, epoch: region.body_epochs[id], ..Peer::default() });
        }
        let mut counts = HashMap::new();
        let mut capacity = 0;
        for actor in registry.controlled.actors.values().filter(|a|a.scene == scene) {
            let input = actor.input.as_ref().ok_or("interval history requires complete queued input")?;
            if input.events.len() > MAX_CONTACTS || counts.len() >= MAX_PLAYERS + MAX_ITEMS {
                return Err("interval history participant/event cap".into());
            }
            let peer = peers.get_mut(&region.bodies[&actor.body]).ok_or("interval actor missing body")?;
            peer.actor = actor.id; peer.kind = actor.kind; peer.registration = actor.lease;
            counts.insert(actor.id, input.events.len());
            capacity += MAX_CONTACTS - input.events.len();
        }
        Ok(Self { start: region.time_nanos, end, peers,
            buffer: Mutex::new(EventBuffer { counts, rows: Vec::with_capacity(capacity), terminal_ineligible: std::collections::HashSet::with_capacity(MAX_PLAYERS+MAX_ITEMS), failed: false }) })
    }
    pub(super) fn finish(self, registry: &mut Registry, scene: i64) -> Result<(), String> {
        let mut buffer = self.buffer.into_inner().map_err(|_|"interval contact collector poisoned")?;
        // Callback scheduling is not an ordering guarantee. Preserve actual step brackets,
        // then deterministically order events within that bracket without inventing CCD times.
        buffer.rows.sort_unstable();
        for (id, event) in buffer.rows {
            let actor = registry.controlled.actors.get_mut(&id).ok_or("interval event actor vanished")?;
            if actor.scene != scene { return Err("interval event actor changed scene".into()); }
            actor.input.as_mut().ok_or("interval event input vanished")?.events.push(event);
        }
        for id in buffer.terminal_ineligible {
            registry.controlled.actors.get_mut(&id).ok_or("terminal contact actor vanished")?
                .input.as_mut().ok_or("terminal contact input vanished")?.terminal_supported=false;
        }
        if buffer.failed { return Err("interval contact history overflow or invalid native event; scene retained".into()); }
        Ok(())
    }
}
impl EventHandler for StepEvents {
    fn handle_collision_event(&self, _: &RigidBodySet, _: &ColliderSet, _: CollisionEvent, _: Option<&ContactPair>) {}
    fn handle_contact_force_event(&self, dt: f64, _: &RigidBodySet, colliders: &ColliderSet,
                                  pair: &ContactPair, total_force: f64) {
        let Ok(mut buffer) = self.buffer.lock() else { return; };
        if buffer.failed { return; }
        let Some(first) = colliders.get(pair.collider1) else { buffer.failed=true; return; };
        let Some(second) = colliders.get(pair.collider2) else { buffer.failed=true; return; };
        let Some(a) = first.parent().and_then(|h|self.peers.get(&h)).copied() else { buffer.failed=true; return; };
        let Some(b) = second.parent().and_then(|h|self.peers.get(&h)).copied() else { buffer.failed=true; return; };
        if !dt.is_finite() || dt <= 0. || !total_force.is_finite() || total_force <= 0. {
            buffer.failed=true; return;
        }
        for manifold in &pair.manifolds {
            // Inactive manifolds may retain cached impulses. They were not solver contacts
            // for this callback and must not become historical physical-contact evidence.
            if manifold.data.solver_contacts.is_empty() { continue; }
            // This is the actual normal impulse sum recorded by Rapier for this manifold.
            // It is not a vector net impulse or a force inferred from endpoint displacement.
            let impulse: f64 = manifold.points.iter().map(|p|p.data.impulse).sum();
            if impulse == 0. { continue; }
            if !impulse.is_finite() || impulse < 0. {
                buffer.failed=true; return;
            }
            let point = manifold.data.solver_contacts.iter().map(|p|p.point).sum::<Vec3>()
                / manifold.data.solver_contacts.len() as f64;
            let normal = manifold.data.normal;
            if !point.is_finite() || !normal.is_finite() || (normal.length()-1.).abs()>1e-6 {
                buffer.failed=true; return;
            }
            for (recipient, other, collider, outward) in [(a,b,pair.collider2,-normal),(b,a,pair.collider1,normal)] {
                if recipient.actor == 0 { continue; }
                // Combine-rule precedence on the OTHER collider can override Min. Preserve
                // actual solved coefficients, including contacts gone before the endpoint.
                if manifold.data.solver_contacts.iter().any(|contact|contact.friction!=0. || contact.restitution!=0.) {
                    buffer.terminal_ineligible.insert(recipient.actor);
                }
                let Some(count) = buffer.counts.get_mut(&recipient.actor) else { buffer.failed=true; return; };
                if *count >= MAX_CONTACTS { buffer.failed=true; return; }
                *count += 1;
                let (slot,generation)=collider.into_raw_parts();
                buffer.rows.push((recipient.actor, [self.start,self.end,other.body,other.epoch,
                    slot as i64,generation as i64,other.actor,other.kind,other.registration,
                    bits(point.x),bits(point.y),bits(point.z),bits(outward.x),bits(outward.y),bits(outward.z),bits(impulse)]));
            }
        }
    }
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
pub(super) fn snapshot_scene(state:&State,scene:i64)->State {
    State { next_lease:state.next_lease,actors:state.actors.iter().filter(|(_,a)|a.scene==scene)
        .map(|(id,a)|(*id,a.clone())).collect() }
}
pub(super) fn publish_scene(state:&mut State,scene:i64,candidate:State) {
    state.actors.retain(|_,a|a.scene!=scene);
    state.next_lease=state.next_lease.max(candidate.next_lease);
    state.actors.extend(candidate.actors);
}
pub(super) fn preview_complete(registry:&Registry,scene:i64)->Result<(),String> {
    let end=transfer::lookup(registry,scene)?.time_nanos;
    for actor in registry.controlled.actors.values().filter(|a|a.scene==scene) {
        if actor.result.is_none()||!actor.input.as_ref().is_some_and(|i|i.started&&i.end==end) {
            return Err("staged seal requires every controlled result at the exact interval end".into());
        }
    }Ok(())
}
pub(super) fn mapped_actor_words(
    state: &State,
    scene: i64,
    bodies: &std::collections::HashSet<i64>,
) -> Vec<i64> {
    let mut selected: Vec<_> = state
        .actors
        .values()
        .filter(|a| a.scene == scene && bodies.contains(&a.body))
        .collect();
    selected.sort_by_key(|a| a.id);
    selected
        .into_iter()
        .flat_map(|a| [a.id, a.kind, a.lease, a.last_sequence, a.body])
        .collect()
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
/// Narrow read-only authority seam for candidate feet-anchored PLAYER shape operations.
/// Caller still owns staged-scene admission and the actual validated native mutation.
pub(super) fn pose_actor(registry: &Registry, scene: i64, ids: &[i64]) -> Result<(i64, f64, Vec<i64>), String> {
    if ids.len() != 7 { return Err("anchored pose requires exact actor lease".into()); }
    let a = actor(registry, scene, ids)?;
    if a.kind != 1 || a.input.is_some() || a.result.is_some() {
        return Err("anchored pose requires idle exact PLAYER actor".into());
    }
    Ok((a.body, a.mass, identity(transfer::lookup(registry, scene)?, a)))
}
pub(super) fn pose_publish_identity(registry: &Registry, scene: i64, id: i64) -> Result<Vec<i64>, String> {
    let a = registry.controlled.actors.get(&id).ok_or("anchored pose actor absent")?;
    if a.scene != scene || a.kind != 1 || a.input.is_some() || a.result.is_some() {
        return Err("anchored pose actor owner or pending state changed".into());
    }
    Ok(identity(transfer::lookup(registry, scene)?, a))
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
        let segment = if input.segments.is_empty() { 0 } else {
            input.segments.iter().position(|segment| segment.end > region.time_nanos)
                .ok_or("segmented input has no remaining motor interval")?
        };
        if !input.segments.is_empty() && next > input.segments[segment].end {
            return Err("native step crosses a controlled motor segment boundary".into());
        }
        let boundary = !input.started || segment != input.active_segment;
        if boundary {
            if input.started && (segment != input.active_segment + 1
                || region.time_nanos != input.segments[input.active_segment].end) {
                return Err("controlled motor skipped its exact boundary".into());
            }
            let drive = if input.segments.is_empty() { input.velocity } else { input.segments[segment].velocity };
            let delta = if input.started { drive - input.segments[input.active_segment].velocity } else { Vec3::ZERO };
            // First drive is total desired velocity. Later drives preserve this input's
            // external response. The delta itself may be 640 m/s for opposite legal drives.
            let velocity = if input.started { body.linvel() + delta } else { drive };
            if !velocity.is_finite() || velocity.length() > MAX_SPEED {
                return Err("segmented motor plus retained reaction exceeds admitted speed".into());
            }
            start.push((a.id, a.body, velocity, delta, segment, !input.started));
        }
    }
    // No motor is written until every actor's next boundary and velocity is admitted.
    let region = registry.scenes.get_mut(&scene).unwrap();
    for (id, key, velocity, delta, segment, first) in start {
        let body = &mut region.sim.rigid_body_set[region.bodies[&key]];
        body.set_linvel(velocity, true);
        if first { body.set_angvel(Vec3::ZERO, true); }
        body.planetary_refresh_motor_predictions();
        let input = registry.controlled.actors.get_mut(&id).unwrap().input.as_mut().unwrap();
        if first { input.initial_velocity = velocity; }
        input.applied_motor_delta += delta;
        input.active_segment = segment;
        input.started = true;
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
        let impulse = (velocity - input.initial_velocity - input.applied_motor_delta) * a.mass;
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
/// The new controller impulse is perpendicular to every actual positive-impulse normal.
/// Actual solver velocity is never projected or interpreted as separable motor/response parts.
fn terminal_projection(actual:Vec3,drive:Vec3,target:Vec3,events:&[[i64;16]]) -> Result<(Vec3,usize),String> {
    if actual.length()>MAX_SPEED || drive.length()>MAX_SPEED || target.length()>MAX_SPEED || events.len()>MAX_CONTACTS {
        return Err("terminal motor speed/contact bound".into());
    }
    let remove=|mut value:Vec3,basis:&[Vec3]| {
        for _ in 0..2 { for axis in basis { value-=*axis*value.dot(*axis); } }value
    };
    let mut basis:Vec<Vec3>=Vec::with_capacity(3);let mut normals=Vec::with_capacity(events.len());
    for event in events {
        let normal=Vec3::new(f64::from_bits(event[12] as u64),f64::from_bits(event[13] as u64),f64::from_bits(event[14] as u64));
        if !normal.is_finite() || (normal.length()-1.).abs()>1e-6 { return Err("invalid retained contact normal".into()); }
        normals.push(normal);
        if basis.len()<3 { let independent=remove(normal,&basis);let length=independent.length();
            if length>1e-12 { basis.push(independent/length); }
        }
    }
    let correction=if basis.len()==3 {Vec3::ZERO}else{remove(target-drive,&basis)};
    if !correction.is_finite() || normals.iter().any(|normal|correction.dot(*normal).abs()>1e-8) {
        return Err("terminal projection failed actual contact constraint".into());
    }
    let after=actual+correction;
    if !after.is_finite() || after.length()>MAX_SPEED { return Err("terminal motor plus retained response exceeds speed envelope".into()); }
    Ok((correction,basis.len()))
}
pub(super) fn dispatch(
    registry: &mut Registry,
    scene: i64,
    op: i32,
    ids: &[i64],
    values: &[f64],
) -> Result<Vec<i64>, String> {
    if registry.transfer.is_some() && !matches!(op, 40 | 42 | 44 | 57 | 58 | 74 | 76 | 77 | 79) {
        return Err("controlled actor mutation forbidden during transfer staging".into());
    }
    match op {
        78 => {
            if ids.len()!=10 { return Err("terminal motor requires exact complete input receipt".into()); }
            require(values,6)?;
            let a=actor(registry,scene,ids)?;
            let region=transfer::lookup(registry,scene)?;
            let input=a.input.as_ref().ok_or("terminal motor input absent")?;
            let result=a.result.as_ref().ok_or("terminal motor requires completed result")?;
            if a.kind!=1 || !input.terminal_supported || result[11..14]!=ids[7..10] || input.end!=region.time_nanos {
                return Err("terminal motor requires exact completed PLAYER input".into());
            }
            let expected=vec(values,0)?;let target=vec(values,3)?;
            if let Some(receipt)=&input.terminal {
                if receipt[14..17]!=vector_bits(expected) || receipt[20..23]!=vector_bits(target)
                    || receipt[26..29]!=vector_bits(actor_body(region,a)?.linvel()) {
                    return Err("terminal motor replay differs from retained original request/state".into());
                }
                return Ok(receipt.clone());
            }
            let body=actor_body(region,a)?;
            if body.linvel()!=expected || result[21..24]!=vector_bits(expected) {
                return Err("terminal motor actual velocity CAS mismatch".into());
            }
            // This controller policy assumes the actual actor collider contract, not a
            // guessed friction impulse decomposition from endpoint velocities.
            if body.colliders().len()!=1 { return Err("terminal motor actor collider shape changed".into()); }
            let collider=&region.sim.collider_set[body.colliders()[0]];
            if collider.friction()!=0. || collider.friction_combine_rule()!=CoefficientCombineRule::Min
                || collider.restitution()!=0. || collider.restitution_combine_rule()!=CoefficientCombineRule::Min {
                return Err("terminal motor requires retained frictionless actor contract".into());
            }
            let drive=input.segments.last().map_or(input.velocity,|segment|segment.velocity);
            let (correction,rank)=terminal_projection(expected,drive,target,&input.events)?;
            let after=expected+correction;
            let next_mutation=region.mutation.checked_add(1).ok_or("terminal motor mutation exhausted")?;
            let key=a.body;
            let mut receipt=result[..14].to_vec();
            receipt.extend(vector_bits(expected));receipt.extend(vector_bits(drive));receipt.extend(vector_bits(target));
            receipt.extend(vector_bits(correction));receipt.extend(vector_bits(after));receipt.push(rank as i64);
            let region=registry.scenes.get_mut(&scene).unwrap();
            let body=&mut region.sim.rigid_body_set[region.bodies[&key]];
            body.set_linvel(after,true);body.planetary_refresh_motor_predictions();
            region.mutation=next_mutation;
            let a=registry.controlled.actors.get_mut(&ids[0]).unwrap();
            a.result.as_mut().unwrap()[21..24].copy_from_slice(&vector_bits(after));
            let input=a.input.as_mut().unwrap();input.applied_motor_delta+=correction;input.terminal=Some(receipt.clone());
            // Existing netImpulse remains external momentum; the explicit terminal motor
            // correction belongs to the controller and is recorded separately above.
            Ok(receipt)
        }
        79 => {
            if ids.len()!=10 { return Err("terminal receipt lookup requires exact input".into()); }
            require(values,0)?;let a=actor(registry,scene,ids)?;
            let result=a.result.as_ref().ok_or("terminal result absent")?;
            if result[11..14]!=ids[7..10] { return Err("terminal lookup input differs".into()); }
            Ok(a.input.as_ref().ok_or("terminal input absent")?.terminal.clone().unwrap_or_default())
        }
        74 => {
            if !ids.is_empty() { return Err("segmented capability takes no identities".into()); }
            require(values, 0)?;
            Ok(vec![1, MAX_MOTOR_SEGMENTS as i64, 53, 136, bits(MAX_SPEED)])
        }
        75 => {
            if ids.len() < 12 || ids[10] < 1 || ids[10] > MAX_MOTOR_SEGMENTS as i64
                || ids.len() != 11 + ids[10] as usize {
                return Err("segmented input requires bounded complete motor intervals".into());
            }
            let count = ids[10] as usize;
            require(values, 10 + 3 * count)?;
            let a = actor(registry, scene, ids)?;
            let region = transfer::lookup(registry, scene)?;
            let pose = transfer::pose(values)?;
            let expected_velocity = vec(values, 7)?;
            let duration = ids[9].checked_sub(ids[8]).ok_or("segmented duration overflow")?;
            if a.input.is_some() || a.result.is_some() || ids[7] <= a.last_sequence
                || ids[8] != region.time_nanos
                || ![12_500_000, 25_000_000, 50_000_000].contains(&duration) {
                return Err("segmented input stale, pending or outside fixed interval".into());
            }
            let body = actor_body(region, a)?;
            if !same_pose(body.position(), &pose) || body.linvel() != expected_velocity {
                return Err("segmented input initial actual pose/velocity CAS mismatch".into());
            }
            let mut segments = Vec::with_capacity(count);
            let mut previous = ids[8];
            for index in 0..count {
                let end = ids[11 + index];
                let velocity = vec(values, 10 + index * 3)?;
                if end <= previous || end > ids[9] || velocity.length() > MAX_SPEED {
                    return Err("segmented input interval ordering or motor speed bound".into());
                }
                segments.push(MotorSegment { end, velocity });
                previous = end;
            }
            if previous != ids[9] { return Err("segmented input does not cover exact full interval".into()); }
            let velocity = segments[0].velocity;
            registry.controlled.actors.get_mut(&ids[0]).unwrap().input = Some(Input {
                sequence: ids[7], start: ids[8], end: ids[9], velocity,
                started: false, initial_velocity: velocity, segments, active_segment: 0,
                applied_motor_delta: Vec3::ZERO, terminal: None, terminal_supported: true, events: Vec::with_capacity(MAX_CONTACTS),
            });
            Ok(vec![ids[7], ids[8], ids[9]])
        }
        76 => {
            if !ids.is_empty() { return Err("next motor boundary takes no identities".into()); }
            require(values, 0)?;
            let now = transfer::lookup(registry, scene)?.time_nanos;
            let next = registry.controlled.actors.values().filter(|a| a.scene == scene)
                .filter_map(|a| a.input.as_ref()).filter_map(|input| {
                    if input.segments.is_empty() { (input.end > now).then_some(input.end) }
                    else { input.segments.iter().find(|segment| segment.end > now).map(|segment| segment.end) }
                }).min().unwrap_or(now);
            Ok(vec![now, next])
        }
        77 => {
            if ids.len() != 7 { return Err("motor state requires exact actor lease".into()); }
            require(values, 0)?;
            let a = actor(registry, scene, ids)?;
            let region = transfer::lookup(registry, scene)?;
            let body = actor_body(region, a)?;
            let pose = body.position();
            let mut out = identity(region, a);
            out.extend(vector_bits(pose.translation));
            out.extend([bits(pose.rotation.x),bits(pose.rotation.y),bits(pose.rotation.z),bits(pose.rotation.w)]);
            out.extend(vector_bits(body.linvel()));
            out.extend([a.input.is_some() as i64,a.result.is_some() as i64]);
            Ok(out)
        }
        57 => {
            if !ids.is_empty() { return Err("interval contact capability takes no identities".into()); }
            require(values,0)?;
            Ok(vec![1,MAX_CONTACTS as i64,(MAX_PLAYERS+MAX_ITEMS) as i64,16])
        }
        58 => {
            if ids.len()!=10 { return Err("interval contact history requires exact result receipt".into()); }
            require(values,0)?;
            let a=actor(registry,scene,ids)?;
            let result=a.result.as_ref().ok_or("completed interval history absent or acknowledged")?;
            if result[11..14]!=ids[7..10] { return Err("interval history result mismatch".into()); }
            let input=a.input.as_ref().ok_or("interval history input absent")?;
            let mut out=result[..14].to_vec();out.push(input.events.len() as i64);
            for event in &input.events { out.extend(event); }
            Ok(out)
        }
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
            transfer::admit_structural(registry,1,1,0)?;
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
                    .active_events(ActiveEvents::CONTACT_FORCE_EVENTS)
                    .contact_force_event_threshold(0.)
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
            region
                .body_history
                .insert(key, time::BodyHistory::allocated(scene, region.time_nanos));
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
                segments: Vec::new(),
                active_segment: 0,
                applied_motor_delta: Vec3::ZERO,
                terminal: None,
                terminal_supported: true,
                events: Vec::with_capacity(MAX_CONTACTS),
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
            if ids.len() != 10 && ids.len() != 13 {
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
            match &a.input.as_ref().ok_or("acknowledged input absent")?.terminal {
                Some(receipt) if ids.len()==13 && ids[10..13]==receipt[20..23] => {},
                None if ids.len()==10 => {},
                _ => return Err("terminal-adjusted result requires its exact terminal-aware ACK".into()),
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
            region.body_history.remove(&key);
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
        49 => {
            if !ids.is_empty() {
                return Err("controlled quiescence takes no caller identities".into());
            }
            require(values, 0)?;
            let region = transfer::lookup(registry, scene)?;
            let mut actors: Vec<_> = registry
                .controlled
                .actors
                .values()
                .filter(|a| a.scene == scene)
                .collect();
            actors.sort_by_key(|a| a.id);
            if actors.len() > MAX_PLAYERS + MAX_ITEMS {
                return Err("controlled quiescence actor cap".into());
            }
            // Clock, membership, pending flags and full current states are captured under this
            // one registry lock. A Java sequence of identity/state calls is not this receipt.
            let mut result = vec![
                1,
                scene,
                region.epoch,
                region.time_nanos,
                region.mutation,
                actors.len() as i64,
                character::pending_in_scene(&registry.characters, scene) as i64,
            ];
            for a in actors {
                let state = transfer::body_state(region, a.body, region.body_epochs[&a.body])?;
                if state.len() != 68 {
                    return Err("controlled quiescence requires one owned box collider".into());
                }
                result.extend(identity(region, a));
                result.extend([
                    a.input.is_some() as i64,
                    a.result.is_some() as i64,
                    state.len() as i64,
                ]);
                result.extend(state.into_iter().map(bits));
            }
            Ok(result)
        }
        _ => Err("unknown controlled actor operation".into()),
    }
}
