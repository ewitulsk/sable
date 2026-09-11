//! Bounded f64 character contact witnesses. No actor rigid-body proxy, force input, or physics step.
use super::*;
use rapier3d_f64::control::{CharacterCollision,CharacterLength,KinematicCharacterController};
use rapier3d_f64::parry::{bounding_volume::BoundingVolume,partitioning::{Bvh,BvhBuildStrategy},query::{DefaultQueryDispatcher,PersistentQueryDispatcher,ContactManifold}};
use std::collections::HashSet;

const MAX_ACTORS:usize=128;
const MAX_WITNESSES:usize=8;
const MAX_QUERY_COLLIDERS:usize=8192;
const MAX_CANDIDATES:usize=64;
const MAX_PRIMITIVES:usize=16384;
const MAX_CONTACTS:usize=64;
const SKIN:f64=0.00001;
#[derive(Clone,PartialEq)]
struct Actor {lease:i64,scene:i64,half:Vec3,mass:f64,last_sequence:i64,last_time:i64,next_time:i64,
    history_domain:i64,history_duration:i64,pending:Option<i64>}
struct Reaction {id:i64,lease:[i64;5],handle:RigidBodyHandle,after:RigidBody}
struct Witness {actor:i64,lease:i64,scene:i64,epoch:i64,time:i64,end_time:i64,mutation:i64,sequence:i64,reactions:Vec<Reaction>}
#[derive(Default)]
pub(super) struct State {next_lease:i64,next_witness:i64,actors:HashMap<i64,Actor>,witnesses:HashMap<i64,Witness>}
pub(super) fn count(state:&State)->usize{state.actors.len()}
pub(super) fn contains(state:&State,id:i64)->bool{state.actors.contains_key(&id)}
pub(super) fn owns_scene(state:&State,scene:i64)->bool{state.actors.values().any(|a|a.scene==scene)}
pub(super) fn pending_in_scene(state:&State,scene:i64)->usize{state.witnesses.values().filter(|w|w.scene==scene).count()}
pub(super) struct MappedActor {id:i64,before:Actor,after:Actor}
pub(super) fn prepare_mapped(registry:&Registry,source:i64,destination:i64,descriptors:&[i64])->Result<Vec<MappedActor>,String>{
    if descriptors.len()%5!=0||descriptors.len()/5>MAX_ACTORS{return Err("legacy clock mapping capacity".into());}
    if pending_in_scene(&registry.characters,source)!=0||pending_in_scene(&registry.characters,destination)!=0{return Err("legacy witness retains clock mapping".into());}
    if !descriptors.is_empty()&&controlled::owns_scene(&registry.controlled,destination){return Err("legacy/controlled mode mixing at mapped destination".into());}
    let source_now=transfer::lookup(registry,source)?.time_nanos;let destination_now=transfer::lookup(registry,destination)?.time_nanos;
    let mut selected=HashSet::new();let mut result=Vec::new();
    for d in descriptors.chunks_exact(5){
        let before=registry.characters.actors.get(&d[0]).ok_or("unknown mapped legacy actor")?;
        if !selected.insert(d[0])||before.scene!=source||before.pending.is_some()||[before.lease,before.last_sequence,before.last_time,before.next_time]!=d[1..5]{return Err("stale mapped legacy actor lease/history".into());}
        let mut after=before.clone();after.scene=destination;after.next_time=time::map_eligibility(source_now,destination_now,before.next_time)?;
        result.push(MappedActor{id:d[0],before:before.clone(),after});
    }
    result.sort_by_key(|a|a.id);Ok(result)
}
pub(super) fn validate_mapped(state:&State,actors:&[MappedActor])->Result<(),String>{
    for a in actors{if state.actors.get(&a.id)!=Some(&a.before){return Err("mapped legacy actor changed after preparation".into());}}Ok(())
}
pub(super) fn mapped_words(actors:&[MappedActor])->Vec<i64>{
    actors.iter().flat_map(|a|[a.id,a.before.lease,a.before.last_sequence,a.before.history_domain,a.before.last_time,a.before.history_duration,a.before.next_time,a.after.next_time]).collect()
}
pub(super) fn commit_mapped(state:&mut State,actors:Vec<MappedActor>){for a in actors{state.actors.insert(a.id,a.after);}}
pub(super) fn retire_scene(state:&mut State,scene:i64){state.actors.retain(|_,a|a.scene!=scene);state.witnesses.retain(|_,w|w.scene!=scene);}
fn identity(region:&Region,id:i64,actor:&Actor)->Vec<i64>{vec![id,actor.lease,actor.scene,region.epoch,actor.last_sequence,actor.last_time,actor.next_time]}
fn actor<'a>(registry:&'a Registry,scene:i64,ids:&[i64])->Result<&'a Actor,String>{
    if ids.len()<4{return Err("complete character lease required".into());}
    let region=transfer::lookup(registry,scene)?;let a=registry.characters.actors.get(&ids[0]).ok_or("unknown character")?;
    if a.lease!=ids[1]||a.scene!=scene||ids[2]!=scene||ids[3]!=region.epoch{return Err("stale character scene/frame/registration lease".into());}Ok(a)
}
fn bits(v:f64)->i64{v.to_bits() as i64}
fn vector_bits(v:Vec3)->[i64;3]{[bits(v.x),bits(v.y),bits(v.z)]}

/// The transient query index uses current parent poses, including adds/motion edits not yet stepped.
/// Shared shapes are Arc-backed; only bounded resident collider metadata is copied, never world voxels.
fn query_snapshot(region:&Region)->Result<(ColliderSet,Bvh),String>{
    if region.sim.collider_set.len()>MAX_QUERY_COLLIDERS{return Err("character resident collider snapshot cap".into());}
    let mut colliders=region.sim.collider_set.clone();
    for (_,c) in colliders.iter_mut(){if let Some(parent)=c.parent(){c.set_position(*region.sim.rigid_body_set[parent].position()*c.position_wrt_parent().unwrap());}}
    let bvh=Bvh::from_iter(BvhBuildStrategy::Binned,colliders.iter().filter(|(_,c)|c.is_enabled()).map(|(h,c)|(h.into_raw_parts().0 as usize,c.compute_aabb())));
    Ok((colliders,bvh))
}
fn prepare(registry:&Registry,scene:i64,ids:&[i64],values:&[f64],witness_id:i64)->Result<(Witness,Vec<i64>),String>{
    if ids.len()!=8{return Err("character witness requires lease, scene clock, intent sequence and substep nanoseconds".into());}require(values,10)?;
    if controlled::owns_scene(&registry.controlled,scene){return Err("legacy witnesses cannot mutate controlled solver participants".into());}
    let a=actor(registry,scene,ids)?;let region=transfer::lookup(registry,scene)?;
    if a.pending.is_some(){return Err("character already owns a pending witness".into());}
    if ids[4]!=region.time_nanos||ids[5]!=region.mutation{return Err("stale character query clock/mutation".into());}
    if ids[6]<=a.last_sequence||region.time_nanos<a.next_time{return Err("character intent already consumed or overlaps a consumed simulation interval".into());}
    if ![12_500_000,25_000_000,50_000_000].contains(&ids[7]){return Err("unsupported character fixed substep".into());}
    let end_time=region.time_nanos.checked_add(ids[7]).ok_or("character interval overflow")?;
    let dt=ids[7] as f64/1e9;let start=transfer::pose(values)?;let motion=vec(values,7)?;
    if motion.length()>32.*dt{return Err("character speed exceeds server-adapter envelope".into());}
    let shape=SharedShape::cuboid(a.half.x,a.half.y,a.half.z);
    let initial=shape.compute_aabb(&start);let finish=shape.compute_aabb(&Pose{translation:start.translation+motion,rotation:start.rotation});
    let broad=initial.merged(&finish).loosened(0.1);bounded(broad.mins)?;bounded(broad.maxs)?;
    let (colliders,bvh)=query_snapshot(region)?;let dispatcher=DefaultQueryDispatcher;
    let queries=QueryPipeline{dispatcher:&dispatcher,bvh:&bvh,bodies:&region.sim.rigid_body_set,colliders:&colliders,filter:QueryFilter::default()};
    let mut candidates=0;let mut primitives=0;
    for (_,c) in queries.intersect_aabb_conservative(broad){
        candidates+=1;primitives+=c.shape().as_compound().map_or(1,|s|s.shapes().len());
        if candidates>MAX_CANDIDATES||primitives>MAX_PRIMITIVES{return Err("character query candidate/primitive cap".into());}
        if rapier3d_f64::parry::query::contact(&start,shape.as_ref(),c.position(),c.shape(),0.).map_err(|_|"unsupported character contact shape")?.is_some_and(|contact|contact.dist < -SKIN){return Err("character witness starts in penetration".into());}
    }
    let controller=KinematicCharacterController{up:start.rotation*Vec3::Y,offset:CharacterLength::Absolute(SKIN),
        autostep:None,snap_to_ground:None,normal_nudge_factor:SKIN,..Default::default()};
    let mut contacts:Vec<CharacterCollision>=Vec::new();let mut overflow=false;
    let movement=controller.move_shape(dt,&queries,shape.as_ref(),&start,motion,|contact|{if contacts.len()<MAX_CONTACTS{contacts.push(contact)}else{overflow=true}});
    if overflow{return Err("character contact witness cap".into());}
    let mut dynamic=HashSet::new();
    for hit in &contacts{if let Some(parent)=colliders[hit.handle].parent(){if region.sim.rigid_body_set[parent].is_dynamic(){dynamic.insert(hit.handle);}}}
    let mut staged=region.sim.rigid_body_set.clone();
    // A wall cannot push an occluded neighbour. One reaction per witnessed dynamic body uses
    // full point effective mass (including rotational inertia), not a repeated mass-only kick
    // per manifold point. The character's finite server-owned mass bounds the momentum exchange.
    let mut reacted=HashSet::new();
    for hit in contacts.iter().filter(|c|dynamic.contains(&c.handle)){
        let handle=colliders[hit.handle].parent().unwrap();if !reacted.insert(handle){continue;}
        let body=&mut staged[handle];let direction=-hit.hit.normal1;
        // The cast witness may select an arbitrary corner of two parallel faces. Use the
        // actual contact patch centroid for its resultant impulse instead of creating an
        // orientation-dependent torque from that arbitrary single GJK witness.
        let current=&colliders[hit.handle];let mut patch:Vec<ContactManifold<(),()>>=Vec::new();
        dispatcher.contact_manifolds(&hit.character_pos.inv_mul(current.position()),shape.as_ref(),current.shape(),SKIN*2.,&mut patch,&mut None).map_err(|_|"unsupported character reaction manifold")?;
        let mut point=Vec3::ZERO;let mut point_count=0;
        for manifold in &patch{let pose=*current.position()*manifold.subshape_pos2.unwrap_or(Pose::IDENTITY);
            for p in &manifold.points{if p.dist<=SKIN*2.{point+=pose*p.local_p2;point_count+=1;if point_count>MAX_CONTACTS{return Err("character reaction manifold cap".into());}}}}
        if point_count==0{return Err("character cast has no current contact patch".into());}point/=point_count as f64;
        let arm=point-body.center_of_mass();let angular=arm.cross(direction);let props=body.mass_properties();
        let inverse_mass=1./a.mass+(direction*props.effective_inv_mass).dot(direction)
            +angular.dot(props.effective_world_inv_inertia*angular);
        let approach=(hit.translation_remaining/dt-body.velocity_at_point(point)).dot(direction).max(0.);
        if !inverse_mass.is_finite()||inverse_mass<=0.||!approach.is_finite(){return Err("invalid character effective contact mass".into());}
        body.apply_impulse_at_point(direction*(approach/inverse_mass),point,true);
    }
    let mut reactions=Vec::new();let mut unique=HashSet::new();
    for collider in dynamic{let handle=colliders[collider].parent().unwrap();if !unique.insert(handle){continue;}
        let id=*region.bodies.iter().find(|(_,h)|**h==handle).ok_or("witness dynamic body has no persistent identity")?.0;
        let after=&staged[handle];bounded(after.linvel())?;bounded(after.angvel())?;
        let (slot,generation)=handle.into_raw_parts();
        reactions.push(Reaction{id,handle,lease:[id,region.body_epochs[&id],slot as i64,generation as i64,region.epoch],after:after.clone()});
    }
    reactions.sort_by_key(|r|r.id);
    let mut response=vec![witness_id,ids[0],a.lease,scene,region.epoch,region.time_nanos,region.mutation,ids[6],ids[7],contacts.len() as i64,reactions.len() as i64];
    response.extend(vector_bits(movement.translation));response.push(movement.grounded as i64);
    for hit in contacts{
        let c=&colliders[hit.handle];let id=c.parent().and_then(|parent|region.bodies.iter().find(|(_,h)|**h==parent).map(|(id,_)|*id)).unwrap_or(0);
        let(slot,generation)=hit.handle.into_raw_parts();response.extend([id,region.body_epochs.get(&id).copied().unwrap_or(0),slot as i64,generation as i64,bits(hit.hit.time_of_impact)]);
        response.extend(vector_bits(hit.hit.witness1));response.extend(vector_bits(hit.hit.normal1));
    }
    Ok((Witness{actor:ids[0],lease:a.lease,scene,epoch:region.epoch,time:region.time_nanos,end_time,mutation:region.mutation,sequence:ids[6],reactions},response))
}
pub(super) fn dispatch(registry:&mut Registry,scene:i64,op:i32,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    match op{
        20=>{if !ids.is_empty(){return Err("character capability takes no identities".into());}require(values,0)?;Ok(vec![1,MAX_ACTORS as i64,MAX_WITNESSES as i64,MAX_CANDIDATES as i64,MAX_CONTACTS as i64])},
        21=>{
            if ids.len()!=1||ids[0]<=0{return Err("positive server-owned character ID required".into());}require(values,4)?;
            let half=vec(values,0)?;if half.min_element()<0.05||half.max_element()>2.||!values[3].is_finite()||values[3]<1.||values[3]>500.{return Err("invalid server-owned character shape/mass".into());}
            transfer::lookup(registry,scene)?;
            if registry.characters.actors.len()+controlled::count(&registry.controlled)>=MAX_ACTORS||registry.characters.actors.contains_key(&ids[0])||(controlled::contains(&registry.controlled,ids[0])||controlled::owns_scene(&registry.controlled,scene)){return Err("character registration capacity or duplicate identity".into());}
            let lease=registry.characters.next_lease.checked_add(1).ok_or("character registration identity exhausted")?;
            let a=Actor{lease,scene,half,mass:values[3],last_sequence:0,last_time:-1,next_time:0,history_domain:scene,history_duration:0,pending:None};let result=identity(&registry.scenes[&scene],ids[0],&a);
            registry.characters.actors.insert(ids[0],a);registry.characters.next_lease=lease;Ok(result)
        },
        22=>{if ids.len()!=1{return Err("character identity needs actor ID".into());}require(values,0)?;let a=registry.characters.actors.get(&ids[0]).ok_or("unknown character")?;if a.scene!=scene{return Err("foreign character scene".into());}Ok(identity(transfer::lookup(registry,scene)?,ids[0],a))},
        23=>{if ids.len()!=4{return Err("character retirement needs complete lease".into());}require(values,0)?;if actor(registry,scene,ids)?.pending.is_some(){return Err("abort pending character witness before retirement".into());}registry.characters.actors.remove(&ids[0]);Ok(vec![])},
        24=>{
            if registry.characters.witnesses.len()>=MAX_WITNESSES{return Err("character witness capacity exhausted".into());}
            let id=registry.characters.next_witness.checked_add(1).ok_or("character witness identity exhausted")?;
            let (witness,result)=prepare(registry,scene,ids,values,id)?;
            registry.characters.actors.get_mut(&ids[0]).unwrap().pending=Some(id);registry.characters.witnesses.insert(id,witness);registry.characters.next_witness=id;Ok(result)
        },
        25=>{
            if ids.len()!=3{return Err("commit needs witness, actor and registration identity".into());}require(values,0)?;
            let w=registry.characters.witnesses.get(&ids[0]).ok_or("retired character witness")?;
            if w.scene!=scene||w.actor!=ids[1]||w.lease!=ids[2]{return Err("foreign character witness".into());}
            let region=transfer::lookup(registry,scene)?;
            if region.epoch!=w.epoch||region.time_nanos!=w.time||region.mutation!=w.mutation{return Err("stale character witness scene/frame/time/mutation".into());}
            let a=registry.characters.actors.get(&w.actor).ok_or("retired character")?;
            if a.lease!=w.lease||a.pending!=Some(ids[0])||a.last_sequence>=w.sequence||w.time<a.next_time{return Err("stale or replayed character intent".into());}
            for r in &w.reactions{if transfer::body_lease(region,&r.lease)?!=r.handle{return Err("stale character body witness".into());}}
            let mutation=region.mutation.checked_add(1).ok_or("scene mutation exhausted")?;
            let w=registry.characters.witnesses.remove(&ids[0]).unwrap();let count=w.reactions.len();
            let region=registry.scenes.get_mut(&scene).unwrap();
            for r in w.reactions{region.sim.rigid_body_set.get_mut(r.handle).unwrap();region.sim.rigid_body_set.planetary_restore_imported_body(r.handle,&r.after,Vec3::ZERO);}
            region.mutation=mutation;
            let a=registry.characters.actors.get_mut(&w.actor).unwrap();a.pending=None;a.last_sequence=w.sequence;a.last_time=w.time;a.next_time=w.end_time;a.history_domain=w.scene;a.history_duration=w.end_time-w.time;
            Ok(vec![ids[0],w.sequence,w.time,w.end_time,count as i64])
        },
        26=>{if ids.len()!=3{return Err("abort needs witness, actor and registration identity".into());}require(values,0)?;let w=registry.characters.witnesses.get(&ids[0]).ok_or("retired character witness")?;if w.scene!=scene||w.actor!=ids[1]||w.lease!=ids[2]{return Err("foreign character witness".into());}registry.characters.witnesses.remove(&ids[0]);registry.characters.actors.get_mut(&ids[1]).unwrap().pending=None;Ok(vec![])},
        27=>{require(values,0)?;Ok(vec![registry.characters.actors.len() as i64,registry.characters.witnesses.len() as i64])},
        _=>Err("unknown character operation".into())
    }
}
