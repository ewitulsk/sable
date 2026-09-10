//! Existing resident-section replacement, one synchronous native commit and bounded replay receipt.
use super::*;
use std::collections::HashSet;
const MAX_BATCH:usize=8;
struct Receipt { request:Vec<i64>,values:Vec<u64>,result:Vec<i64> }
#[derive(Default)]
pub(super) struct State { high_water:i64,receipt:Option<Receipt> }

pub(super) fn dispatch(registry:&mut Registry,scene:i64,op:i32,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if op==30{if !ids.is_empty(){return Err("terrain batch capability takes no identities".into());}require(values,0)?;return Ok(vec![1,MAX_BATCH as i64]);}
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
    if op!=31||ids.len()<6||ids[4]<=0||ids[5]<1||ids[5]>MAX_BATCH as i64{return Err("invalid bounded terrain batch header".into());}
    let count=ids[5] as usize;if ids.len()!=6+count*6{return Err("incomplete terrain batch identities".into());}require(values,count*3)?;
    let region=transfer::lookup(registry,scene)?;
    let value_bits:Vec<u64>=values.iter().map(|v|v.to_bits()).collect();
    if let Some(receipt)=&region.terrain_batch.receipt{
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
    let mut wake=HashSet::new();
    for (i,entry) in ids[6..].chunks_exact(6).enumerate(){
        if !leases.insert(entry[0])||!geometries.insert(entry[5]){return Err("duplicate terrain lease or prepared geometry".into());}
        let old=region.sections.get(&entry[0]).ok_or("retired terrain batch lease")?;
        if !old.resident||entry[3]!=1||old.revision!=entry[1]||old.fingerprint as i64!=entry[2]||entry[4]<=old.revision{return Err("stale terrain batch section identity/revision".into());}
        if old.body.is_some()!=old.collider.is_some(){return Err("inconsistent terrain collider ownership".into());}
        if let Some(handle)=old.body{if !region.sim.rigid_body_set.get(handle).is_some_and(|b|b.is_fixed()&&b.colliders()==[old.collider.unwrap()]){return Err("terrain section is not an isolated fixed collider".into());}}
        let shape=prepared.shapes.get(&entry[5]).ok_or("retired prepared geometry")?;
        let position=vec(values,i*3)?;bounded(position+Vec3::splat(16.))?;
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
    let receipt=Receipt{request:ids.to_vec(),values:value_bits,result:result.clone()};
    let region=registry.scenes.get_mut(&scene).unwrap();
    region.sim.rigid_body_set.planetary_reserve_static_batch(count+wake.len());
    region.sim.collider_set.planetary_reserve_static_batch(count);
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
