//! Existing resident-section replacement, one synchronous native commit and bounded replay receipt.
use super::*;
use std::collections::HashSet;
const MAX_BATCH:usize=8;
const MAX_BODY_CANDIDATES:usize=64;
const MAX_CONTACT_PARTS:usize=16384;
const PENETRATION_EPSILON:f64=0.00001;
#[derive(Clone)]
struct Receipt { request:Vec<i64>,values:Vec<u64>,result:Vec<i64> }
#[derive(Clone,Default)]
pub(super) struct State { high_water:i64,receipt:Option<Receipt> }
pub(super) fn idle(state:&State)->bool { state.receipt.is_none() }

fn primitive_key(pose:&Pose,shape:&SharedShape)->Result<[u64;10],String>{
    let half=shape.as_cuboid().ok_or("terrain edit requires compiled box primitives")?.half_extents;
    Ok([pose.translation.x.to_bits(),pose.translation.y.to_bits(),pose.translation.z.to_bits(),
        pose.rotation.x.to_bits(),pose.rotation.y.to_bits(),pose.rotation.z.to_bits(),pose.rotation.w.to_bits(),
        half.x.to_bits(),half.y.to_bits(),half.z.to_bits()])
}
// Inspect real current solver poses, including controlled actors. Only exact unchanged
// primitives inherit an existing contact: an old floor's deeper penetration must not hide
// a newly inserted wall. Changed primitives conservatively refuse current penetration.
fn preflight_bodies(region:&Region,old:&Section,shape:&PreparedShape,position:Vec3,
    candidates:&mut usize,parts:&mut usize)->Result<(),String>{
    if old.fingerprint==shape.fingerprint&&old.translation==position{return Ok(());}
    let Some(proposed)=&shape.shape else{return Ok(());};
    if region.sim.collider_set.len()>8192{return Err("terrain edit collider enumeration cap".into());}
    let pose=transfer::pose(&[position.x,position.y,position.z,0.,0.,0.,1.])?;
    let bounds=proposed.compute_aabb(&pose);
    for (_,collider) in region.sim.collider_set.iter(){
        let Some(parent)=collider.parent() else{continue;};
        let body=&region.sim.rigid_body_set[parent];
        if !collider.is_enabled()||body.is_fixed(){continue;}
        let current=*body.position()*collider.position_wrt_parent().ok_or("terrain edit collider lost parent transform")?;
        if !bounds.intersects(&collider.shape().compute_aabb(&current)){continue;}
        *candidates=candidates.checked_add(1).ok_or("terrain edit candidate overflow")?;
        let actor_parts=collider.shape().as_compound().map_or(1,|c|c.shapes().len());
        let work=shape.budget.0.checked_add(old._parts.0).and_then(|p|p.checked_mul(actor_parts)).ok_or("terrain edit contact work overflow")?;
        *parts=parts.checked_add(work).ok_or("terrain edit contact work overflow")?;
        if *candidates>MAX_BODY_CANDIDATES||*parts>MAX_CONTACT_PARTS{return Err("terrain edit body candidate/primitive cap".into());}
        let mut unchanged=HashSet::new();
        if let Some(handle)=old.collider{
            let original=&region.sim.collider_set[handle];
            let compound=original.shape().as_compound().ok_or("original terrain is not compiled compound geometry")?;
            for (local,primitive) in compound.shapes(){unchanged.insert(primitive_key(&(*original.position()*local),primitive)?);}
        }
        let compound=proposed.as_compound().ok_or("candidate terrain is not compiled compound geometry")?;
        for (local,primitive) in compound.shapes(){
            let world_pose=pose*local;
            if unchanged.contains(&primitive_key(&world_pose,primitive)?){continue;}
            if let Some(contact)=rapier3d_f64::parry::query::contact(&world_pose,primitive.as_ref(),&current,collider.shape(),0.)
                .map_err(|_|"unsupported terrain edit body shape")?{
                if !contact.dist.is_finite()||contact.dist < -PENETRATION_EPSILON{return Err("changed terrain primitive penetrates a physical body".into());}
            }
        }
    }
    Ok(())
}

pub(super) fn dispatch(registry:&mut Registry,scene:i64,op:i32,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if op==30{if !ids.is_empty(){return Err("terrain batch capability takes no identities".into());}require(values,0)?;return Ok(vec![2,MAX_BATCH as i64]);}
    if op==35{if !ids.is_empty(){return Err("body-aware terrain capability takes no identities".into());}require(values,0)?;return Ok(vec![1,MAX_BODY_CANDIDATES as i64,MAX_CONTACT_PARTS as i64]);}
    if op==32||op==33{
        if ids.len()!=1||ids[0]<=0{return Err("terrain batch action ID required".into());}require(values,0)?;
        let region=registry.scenes.get_mut(&scene).ok_or("stale terrain batch scene")?;
        let Some(receipt)=&region.terrain_batch.receipt else {
            if op==33&&ids[0]==region.terrain_batch.high_water{return Ok(vec![]);}
            if ids[0]<=region.terrain_batch.high_water{return Err("terrain batch action already acknowledged".into());}
            return if op==32{Ok(vec![])}else{Err("unknown terrain batch acknowledgement".into())};
        };
        if receipt.request[4]!=ids[0]{return Err("another terrain batch receipt is outstanding".into());}
        if op==32{return Ok(receipt.result.clone());}
        region.terrain_batch.receipt=None;return Ok(vec![]);
    }
    if ![31,34].contains(&op)||ids.len()<6||ids[4]<=0||ids[5]<1||ids[5]>MAX_BATCH as i64{return Err("invalid bounded terrain batch header".into());}
    let count=ids[5] as usize;if ids.len()!=6+count*6{return Err("incomplete terrain batch identities".into());}require(values,count*3)?;
    let region=transfer::lookup(registry,scene)?;
    let value_bits:Vec<u64>=values.iter().map(|v|v.to_bits()).collect();
    if let Some(receipt)=&region.terrain_batch.receipt{
        if op==34{return Err("terrain batch was already committed before preflight".into());}
        if receipt.request==ids&&receipt.values==value_bits{return Ok(receipt.result.clone());}
        return Err("unacknowledged terrain batch or conflicting action replay".into());
    }
    if ids[4]<=region.terrain_batch.high_water{return Err("retired terrain batch action ID".into());}
    if ids[..4]!=[scene,region.epoch,region.time_nanos,region.mutation]{return Err("stale terrain batch scene clock".into());}
    if registry.transfer.is_some(){return Err("terrain publication cannot overlap prepared group transfer".into());}
    if !region.streamed_terrain{return Err("terrain batch requires existing streamed leases".into());}
    let mutation=region.mutation.checked_add(1).ok_or("scene mutation exhausted")?;
    let mut prepared=PREPARED.get_or_init(||Mutex::new(PreparedRegistry::default())).lock().map_err(|_|"prepared geometry registry poisoned")?;
    let mut leases=HashSet::with_capacity(count);let mut geometries=HashSet::with_capacity(count);
    let mut positions=Vec::with_capacity(count);let mut result=Vec::with_capacity(7+count*4);
    result.extend([scene,region.epoch,region.time_nanos,region.mutation,mutation,ids[4],count as i64]);
    let mut old_parts=0usize;let mut new_parts=0usize;
    let mut new_instances=0usize;
    let mut wake=HashSet::new();
    let mut body_candidates=0usize;let mut contact_parts=0usize;
    for (i,entry) in ids[6..].chunks_exact(6).enumerate(){
        if !leases.insert(entry[0])||!geometries.insert(entry[5]){return Err("duplicate terrain lease or prepared geometry".into());}
        let old=region.sections.get(&entry[0]).ok_or("retired terrain batch lease")?;
        if !old.resident||entry[3]!=1||old.revision!=entry[1]||old.fingerprint as i64!=entry[2]||entry[4]<=old.revision{return Err("stale terrain batch section identity/revision".into());}
        if old.body.is_some()!=old.collider.is_some(){return Err("inconsistent terrain collider ownership".into());}
        if let Some(handle)=old.body{if !region.sim.rigid_body_set.get(handle).is_some_and(|b|b.is_fixed()&&b.colliders()==[old.collider.unwrap()]){return Err("terrain section is not an isolated fixed collider".into());}}
        let shape=prepared.shapes.get(&entry[5]).ok_or("retired prepared geometry")?;
        if old.body.is_none()&&shape.shape.is_some(){new_instances+=1;}
        let position=vec(values,i*3)?;bounded(position+Vec3::splat(16.))?;
        preflight_bodies(region,old,shape,position,&mut body_candidates,&mut contact_parts)?;
        if old.fingerprint!=shape.fingerprint||old.translation!=position{
            if let Some(collider)=old.collider{for pair in region.sim.narrow_phase.contact_pairs_with(collider){
                let other=if pair.collider1==collider{pair.collider2}else{pair.collider1};
                if let Some(parent)=region.sim.collider_set.get(other).and_then(|c|c.parent()){
                    if region.sim.rigid_body_set[parent].is_dynamic(){wake.insert(parent);if wake.len()>512{return Err("terrain edit affected-body cap".into());}}
                }
            }}
        }
        old_parts=old_parts.checked_add(old._parts.0).ok_or("terrain part count overflow")?;
        new_parts=new_parts.checked_add(shape.budget.0).ok_or("terrain part count overflow")?;
        positions.push(position);result.extend([entry[0],entry[4],shape.fingerprint as i64,1]);
    }
    // Prepared shapes already own their global reservations. Replacement moves those budgets;
    // no second allocation/reservation is charged for their installation.
    let live=LIVE_PARTS.load(Ordering::Acquire);
    if old_parts>live||new_parts>MAX_TOTAL_PARTS||live-old_parts>MAX_TOTAL_PARTS||region.sections.len()>MAX_SECTIONS{return Err("terrain replacement net capacity exceeded".into());}
    // Existing nonempty replacements retain their actual body/collider identities. Empty-to-
    // nonempty rows allocate; charge every such row before any removal or prepared consumption.
    transfer::admit_structural(registry,new_instances,new_instances,0)?;
    let receipt=Receipt{request:ids.to_vec(),values:value_bits,result:result.clone()};
    let region=registry.scenes.get_mut(&scene).unwrap();
    region.sim.rigid_body_set.planetary_reserve_static_batch(count+wake.len());
    region.sim.collider_set.planetary_reserve_static_batch(count);
    if op==34{return Ok(vec![scene,region.epoch,region.time_nanos,region.mutation,count as i64]);}
    // All fallible validation and bounded bookkeeping allocation precede authoritative mutation.
    // A panic poisons the registry; callers retain their fence and must not claim rollback.
    for (i,entry) in ids[6..].chunks_exact(6).enumerate(){
        let shape=prepared.shapes.remove(&entry[5]).unwrap();let section=region.sections.get_mut(&entry[0]).unwrap();let sim=&mut region.sim;
        match (&shape.shape,section.body,section.collider){
            (Some(new_shape),Some(body),Some(collider))=>{
                sim.collider_set[collider].set_shape(new_shape.clone());
                sim.rigid_body_set[body].set_translation(positions[i],false);
                sim.collider_set[collider].set_position(*sim.rigid_body_set[body].position());
            },
            (Some(new_shape),None,None)=>{
                let body=sim.rigid_body_set.insert(RigidBodyBuilder::fixed().translation(positions[i]));
                let collider=sim.collider_set.insert_with_parent(ColliderBuilder::new(new_shape.clone()).density(0.).friction(0.45).build(),body,&mut sim.rigid_body_set);
                section.body=Some(body);section.collider=Some(collider);
            },
            (None,Some(body),Some(_))=>{
                sim.rigid_body_set.remove(body,&mut sim.island_manager,&mut sim.collider_set,&mut sim.impulse_joint_set,&mut sim.multibody_joint_set,true);
                section.body=None;section.collider=None;
            },
            (None,None,None)=>{},_=>unreachable!(),
        }
        section.revision=entry[4];section.translation=positions[i];section.fingerprint=shape.fingerprint;section._parts=shape.budget;
    }
    for body in wake{region.sim.rigid_body_set[body].wake_up(true);}
    region.mutation=mutation;region.terrain_batch.high_water=ids[4];region.terrain_batch.receipt=Some(receipt);Ok(result)
}
