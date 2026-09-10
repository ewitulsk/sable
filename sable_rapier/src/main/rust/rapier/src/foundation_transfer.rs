//! Atomic bounded group transfer. Staging never advances simulation time.
use super::*;
use std::collections::{HashMap,HashSet};
use jni::objects::JLongArray;
use jni::sys::jlongArray;

const MAX_BODY_INSTANCES: usize = 16384;
const MAX_COLLIDER_INSTANCES: usize = 32768;
const MAX_JOINT_INSTANCES: usize = 16384;
const MAX_CONTACT_PAIRS: usize = 32768;
const MAX_MANIFOLDS: usize = 65536;
const MAX_CONTACT_POINTS: usize = 262144;

#[derive(Clone,Copy,Default)]
struct Resources { bodies: usize, colliders: usize, joints: usize, pairs: usize, manifolds: usize, points: usize }
impl Resources {
    fn scene(sim:&Simulation)->Self {
        let mut result=Self { bodies:sim.rigid_body_set.len(),colliders:sim.collider_set.len(),
            joints:sim.impulse_joint_set.len(),..Self::default() };
        for pair in sim.narrow_phase.contact_pairs() {
            result.pairs+=1; result.manifolds+=pair.manifolds.len();
            result.points+=pair.manifolds.iter().map(|m|m.points.len()+m.data.solver_contacts.len()).sum::<usize>();
        }
        result
    }
    fn add(&mut self,other:Self) {
        self.bodies+=other.bodies; self.colliders+=other.colliders; self.joints+=other.joints;
        self.pairs+=other.pairs; self.manifolds+=other.manifolds; self.points+=other.points;
    }
    fn bounded(self)->Result<(),String> {
        if self.bodies>MAX_BODY_INSTANCES || self.colliders>MAX_COLLIDER_INSTANCES
            || self.joints>MAX_JOINT_INSTANCES || self.pairs>MAX_CONTACT_PAIRS
            || self.manifolds>MAX_MANIFOLDS || self.points>MAX_CONTACT_POINTS {
            return Err("transfer staging plus authoritative scene resource cap exceeded".into());
        }
        Ok(())
    }
    fn values(self)->Vec<i64> { vec![self.bodies as i64,self.colliders as i64,self.joints as i64,
        self.pairs as i64,self.manifolds as i64,self.points as i64] }
}

impl Simulation {
    fn initialize_static_indexes(&mut self) {
        // With no actors/constraints there is no solver state to preserve. Rebuilding the indexes
        // also removes retired actors' deferred contacts and acknowledges exact current terrain.
        self.pipeline=PhysicsPipeline::new();self.island_manager=IslandManager::new();
        self.narrow_phase=NarrowPhase::new();self.ccd_solver=CCDSolver::new();
        self.broad_phase=DefaultBroadPhase::new();
        self.broad_phase.planetary_initialize_static(&self.parameters,&self.rigid_body_set,&self.collider_set);
        self.rigid_body_set.planetary_clear_static_changes();self.collider_set.planetary_clear_static_changes();
    }
    fn staged_clone(&self)->Self {
        Self { pipeline:PhysicsPipeline::new(),rigid_body_set:self.rigid_body_set.clone(),
            collider_set:self.collider_set.clone(),island_manager:self.island_manager.clone(),
            broad_phase:self.broad_phase.clone(),narrow_phase:self.narrow_phase.clone(),
            impulse_joint_set:self.impulse_joint_set.clone(),multibody_joint_set:self.multibody_joint_set.clone(),
            ccd_solver:CCDSolver::new(),gravity:self.gravity,parameters:self.parameters }
    }
}

pub(super) struct PreparedTransfer {
    id:i64, source:i64, destination:i64, source_mutation:i64, destination_mutation:i64,
    source_epoch:i64, destination_epoch:i64, time_nanos:i64,
    source_sim:Simulation, destination_sim:Simulation,
    source_bodies:HashMap<i64,RigidBodyHandle>, destination_bodies:HashMap<i64,RigidBodyHandle>,
    source_epochs:HashMap<i64,i64>, destination_epochs:HashMap<i64,i64>,
    source_joints:HashMap<i64,ImpulseJointHandle>, destination_joints:HashMap<i64,ImpulseJointHandle>,
    resources:Resources, moved_bodies:usize, moved_joints:usize, moved_contacts:usize,
}

pub(super) fn retire_scene(registry:&mut Registry,scene:i64) {
    if registry.transfer.as_ref().is_some_and(|t|t.source==scene||t.destination==scene) {
        registry.transfer=None;
    }
}

pub(super) fn lookup<'a>(registry:&'a Registry,id:i64)->Result<&'a Region,String> {
    let scene=registry.scenes.get(&id).ok_or("stale scene handle")?;
    if scene.failed_range { return Err("failed scene cannot transfer".into()); }
    Ok(scene)
}
pub(super) fn body_lease(region:&Region,ids:&[i64])->Result<RigidBodyHandle,String> {
    if ids.len()!=5 || region.failed_range || region.body_epochs.get(&ids[0])!=Some(&ids[1]) || region.epoch!=ids[4] {
        return Err("failed scene or stale body/frame ownership lease".into());
    }
    let handle=*region.bodies.get(&ids[0]).ok_or("unknown body")?;
    let (slot,generation)=handle.into_raw_parts();
    if ids[2]!=slot as i64 || ids[3]!=generation as i64 {return Err("recycled native body lease".into());}
    Ok(handle)
}
/// Conservative full-orientation sweep. End-pose AABBs miss a rotating thin body's middle arc.
/// Pending kinematic motion and forces are included before their first simulation step.
fn swept_bounds(body:&RigidBody,collider:&Collider,gravity:Vec3)->(Vec3,Vec3) {
    let dt=0.05;
    let local=collider.shape().compute_aabb(collider.position_wrt_parent().unwrap());
    let com=body.local_center_of_mass();
    let radius=(local.mins-com).abs().max((local.maxs-com).abs()).length()+collider.contact_skin();
    let current=body.position().translation+body.position().rotation*com;
    let queued=body.next_position().translation+body.next_position().rotation*com;
    let travel=body.linvel().abs()*dt+body.planetary_linear_acceleration(gravity).abs()*dt*dt;
    let halo=Vec3::splat(radius+body.soft_ccd_prediction()+0.05);
    (current.min(queued)-travel-halo,current.max(queued)+travel+halo)
}
pub(super) fn pose(values:&[f64])->Result<Pose,String> {
    let translation=vec(values,0)?;
    if values.len()<7 || values[3..7].iter().any(|v|!v.is_finite()) {return Err("invalid pose rotation".into());}
    let rotation=rapier3d_f64::glamx::DQuat::from_xyzw(values[3],values[4],values[5],values[6]);
    if (rotation.length_squared()-1.).abs()>1e-6{return Err("pose requires unit quaternion".into());}
    Ok(Pose {translation,rotation:rotation.normalize()})
}
pub(super) fn body_state(region:&Region,key:i64,ownership:i64)->Result<Vec<f64>,String> {
    if region.body_epochs.get(&key)!=Some(&ownership){return Err("stale body ownership epoch".into());}
    let rb=&region.sim.rigid_body_set[*region.bodies.get(&key).ok_or("unknown body")?];
    let mut values=Vec::new();
    for p in [rb.position(),rb.next_position()] {values.extend([p.translation.x,p.translation.y,p.translation.z,p.rotation.x,p.rotation.y,p.rotation.z,p.rotation.w]);}
    for v in [rb.linvel(),rb.angvel(),rb.user_force(),rb.user_torque()] {values.extend([v.x,v.y,v.z]);}
    values.push(rb.mass());
    let com=rb.local_center_of_mass();let inertia=rb.mass_properties().local_mprops.principal_inertia();
    values.extend([com.x,com.y,com.z,inertia.x,inertia.y,inertia.z]);
    let frame=rb.mass_properties().local_mprops.principal_inertia_local_frame;values.extend([frame.x,frame.y,frame.z,frame.w]);
    values.extend([rb.linear_damping(),rb.angular_damping(),if rb.is_ccd_enabled(){1.}else{0.},rb.soft_ccd_prediction(),
        rb.locked_axes().bits() as f64,if rb.is_sleeping(){1.}else{0.},rb.activation().normalized_linear_threshold,
        rb.activation().angular_threshold,rb.activation().time_until_sleep,rb.activation().time_since_can_sleep,
        rb.gravity_scale(),rb.dominance_group() as f64,if rb.is_enabled(){1.}else{0.},
        if rb.is_kinematic(){1.}else{0.},rb.colliders().len() as f64]);
    for handle in rb.colliders() {
        let c=&region.sim.collider_set[*handle];let half=c.shape().as_cuboid().ok_or("unsupported exported non-box body collider")?.half_extents;
        let p=c.position_wrt_parent().ok_or("body collider lacks parent")?;
        values.extend([half.x,half.y,half.z,p.translation.x,p.translation.y,p.translation.z,p.rotation.x,p.rotation.y,p.rotation.z,p.rotation.w,
            c.mass(),c.friction(),c.restitution(),c.contact_skin(),c.friction_combine_rule() as u8 as f64,c.restitution_combine_rule() as u8 as f64]);
    }
    Ok(values)
}
fn prepare(registry:&Registry,source_id:i64,args:&[i64],v:&[f64],id:i64)->Result<PreparedTransfer,String> {
    require(v,3)?; let delta=vec(v,0)?;
    if args.len()<5 { return Err("incomplete transfer descriptor".into()); }
    let destination_id=args[0];
    if source_id==destination_id { return Err("transfer requires distinct scenes".into()); }
    let source=lookup(registry,source_id)?; let destination=lookup(registry,destination_id)?;
    if source.epoch!=args[1] || destination.epoch!=args[2] { return Err("stale transfer frame".into()); }
    if source.time_nanos!=destination.time_nanos { return Err("scene simulation clocks differ".into()); }
    if source.sim.gravity!=destination.sim.gravity { return Err("scene gravity mismatch; use common scene field and body overrides".into()); }
    let count=usize::try_from(args[3]).map_err(|_|"invalid body count")?;
    if count==0 || count>4096 || args.len()<5+count { return Err("invalid transfer group size".into()); }
    let ids=&args[4..4+count];
    let links=usize::try_from(args[4+count]).map_err(|_|"invalid collision link count")?;
    if links>4096 || args.len()!=5+count+links*4 { return Err("invalid collision readiness links".into()); }
    if destination.bodies.len()+count>4096 { return Err("destination body capacity exhausted".into()); }
    let mut selected=HashSet::new();
    for stable in ids {
        let body=*source.bodies.get(stable).ok_or("unknown source body")?;
        if !selected.insert(body) { return Err("duplicate transfer body".into()); }
        if destination.bodies.contains_key(stable) { return Err("destination already owns body identity".into()); }
    }
    let mut body_remap=HashMap::new(); let mut collider_remap=HashMap::new(); let mut mapped_sections=HashSet::new();
    for link in args[5+count..].chunks_exact(4) {
        let a=source.sections.get(&link[0]).ok_or("missing source collision lease")?;
        let b=destination.sections.get(&link[2]).ok_or("missing destination collision lease")?;
        if !mapped_sections.insert(link[0]) || !a.resident || !b.resident || a.revision!=link[1] || b.revision!=link[3]
            || a.fingerprint!=b.fingerprint || (a.translation-delta-b.translation).abs().max_element()>1e-9 {
            return Err("stale or mismatched destination collision".into());
        }
        match (a.collider,b.collider,a.body,b.body) {
            (Some(ac),Some(bc),Some(ab),Some(bb)) => {collider_remap.insert(ac,bc);body_remap.insert(ab,bb);},
            (None,None,None,None)=>{},
            _=>return Err("incompatible destination collision topology".into()),
        }
    }
    let mut moved_joints=Vec::new();
    if source.joints.len()!=source.sim.impulse_joint_set.len() || source.sim.multibody_joint_set.iter().next().is_some() {
        return Err("transfer requires complete registered impulse-joint topology".into());
    }
    for (stable,handle) in &source.joints {
        let joint=source.sim.impulse_joint_set.get(*handle).ok_or("stale source joint registry")?;
        let a=selected.contains(&joint.body1); let b=selected.contains(&joint.body2);
        if a!=b { return Err("constraint crosses transfer group boundary".into()); }
        if a {
            if destination.joints.contains_key(stable) { return Err("destination already owns joint identity".into()); }
            moved_joints.push((*stable,joint.clone()));
        }
    }
    if destination.sim.impulse_joint_set.len()+moved_joints.len()>4096 { return Err("destination joint capacity exhausted".into()); }
    let mut contacts=Vec::new();
    for pair in source.sim.narrow_phase.contact_pairs() {
        let a=source.sim.collider_set[pair.collider1].parent();
        let b=source.sim.collider_set[pair.collider2].parent();
        let sa=a.is_some_and(|h|selected.contains(&h)); let sb=b.is_some_and(|h|selected.contains(&h));
        if !sa&&!sb { continue; }
        for other in [a.filter(|_|!sa),b.filter(|_|!sb)].into_iter().flatten() {
            if !source.sim.rigid_body_set[other].is_fixed() && pair.has_any_active_contact() {
                return Err("dynamic contact crosses transfer group boundary".into());
            }
        }
        if (sa||collider_remap.contains_key(&pair.collider1)) && (sb||collider_remap.contains_key(&pair.collider2)) {
            contacts.push(pair);
        } else if pair.has_any_active_contact() { return Err("active contact lacks destination collision mapping".into()); }
    }
    // Reserve the entire swept AABB's existing collision tiles, including explicit air.
    // Missing tiles are unavailable; an empty scene is not a declaration that space is air.
    let base=source.sections.values().find(|s|s.resident).ok_or("source collision not ready")?.translation;
    for handle in &selected {
        let body=&source.sim.rigid_body_set[*handle];
        for collider in body.colliders() {
            let shape=&source.sim.collider_set[*collider];
            let (min,max)=swept_bounds(body,shape,source.sim.gravity);
            bounded(min-delta)?;bounded(max-delta)?;
            for (other_handle,other) in source.sim.rigid_body_set.iter() {
                if selected.contains(&other_handle)||other.is_fixed(){continue;}
                for other_collider in other.colliders() {
                    let c=&source.sim.collider_set[*other_collider];let (other_min,other_max)=swept_bounds(other,c,source.sim.gravity);
                    if min.cmple(other_max).all()&&max.cmpge(other_min).all(){
                        return Err("predicted dynamic interaction crosses transfer group boundary".into());
                    }
                }
            }
            let first=((min-base)/16.).floor().as_ivec3();let last=((max-base)/16.).floor().as_ivec3();
            let size=last-first;
            if (size.x as i64+1)*(size.y as i64+1)*(size.z as i64+1)>4096 { return Err("transfer swept collision footprint exceeds budget".into()); }
            for x in first.x..=last.x { for y in first.y..=last.y { for z in first.z..=last.z {
                let translation=base+Vec3::new(x as f64,y as f64,z as f64)*16.;
                if !source.sections.iter().any(|(key,s)|s.resident && mapped_sections.contains(key)
                    && (s.translation-translation).abs().max_element()<1e-9) {
                    return Err("predicted transfer collision is not ready".into());
                }
            }}}
        }
    }
    let mut live=Resources::default();for region in registry.scenes.values(){live.add(Resources::scene(&region.sim));}
    let mut staging=Resources::scene(&source.sim);staging.add(Resources::scene(&destination.sim));
    let mut peak=live;peak.add(staging);
    // Destination imports temporarily coexist with staged source copies until source removal.
    let extra_colliders=selected.iter().map(|h|source.sim.rigid_body_set[*h].colliders().len()).sum();
    let mut extra=Resources {bodies:count,colliders:extra_colliders,joints:moved_joints.len(),pairs:contacts.len(),..Resources::default()};
    for pair in &contacts {extra.manifolds+=pair.manifolds.len();extra.points+=pair.manifolds.iter().map(|m|m.points.len()+m.data.solver_contacts.len()).sum::<usize>();}
    peak.add(extra);peak.bounded()?;
    let mut source_sim=source.sim.staged_clone();let mut destination_sim=destination.sim.staged_clone();
    let mut source_bodies=source.bodies.clone();let mut destination_bodies=destination.bodies.clone();
    let mut source_epochs=source.body_epochs.clone();let mut destination_epochs=destination.body_epochs.clone();
    let mut source_joints=source.joints.clone();let mut destination_joints=destination.joints.clone();
    for stable in ids {
        let old=source.bodies[stable];let original=&source.sim.rigid_body_set[old];
        let new=destination_sim.rigid_body_set.insert(original.clone()); body_remap.insert(old,new);
        for collider in original.colliders() {
            let new_collider=destination_sim.collider_set.planetary_import_collider(&source.sim.collider_set[*collider],new,&mut destination_sim.rigid_body_set,delta);
            destination_sim.broad_phase.planetary_import_leaf(&source.sim.broad_phase,*collider,new_collider,delta);
            collider_remap.insert(*collider,new_collider);
        }
        destination_sim.rigid_body_set.planetary_restore_imported_body(new,original,delta);
        destination_bodies.insert(*stable,new);source_bodies.remove(stable);
        destination_epochs.insert(*stable,source_epochs.remove(stable).ok_or("missing source ownership epoch")?.checked_add(1).ok_or("ownership epoch exhausted")?);
    }
    let island_remap:Vec<_>=selected.iter().map(|old|(*old,body_remap[old])).collect();
    destination_sim.island_manager.planetary_import_islands(&source.sim.island_manager,&mut destination_sim.rigid_body_set,&island_remap);
    for (stable,original) in &moved_joints {
        let joint=destination_sim.impulse_joint_set.insert(body_remap[&original.body1],body_remap[&original.body2],original.data,false);
        destination_sim.impulse_joint_set.get_mut(joint,false).unwrap().impulses=original.impulses;
        destination_joints.insert(*stable,joint);source_joints.remove(stable);
    }
    for original in &contacts {
        let mut pair=(**original).clone();pair.collider1=collider_remap[&pair.collider1];pair.collider2=collider_remap[&pair.collider2];
        for manifold in &mut pair.manifolds {
            manifold.data.rigid_body1=manifold.data.rigid_body1.map(|old|body_remap[&old]);
            manifold.data.rigid_body2=manifold.data.rigid_body2.map(|old|body_remap[&old]);
            for contact in &mut manifold.data.solver_contacts {contact.point-=delta;}
        }
        destination_sim.narrow_phase.planetary_import_contact(&destination_sim.collider_set,pair);
    }
    for handle in selected {
        source_sim.rigid_body_set.remove(handle,&mut source_sim.island_manager,&mut source_sim.collider_set,
            &mut source_sim.impulse_joint_set,&mut source_sim.multibody_joint_set,true).ok_or("staged source body absent")?;
    }
    source_sim.validate_bounds(Vec3::ZERO)?;destination_sim.validate_bounds(Vec3::ZERO)?;
    let mut resources=Resources::scene(&source_sim);resources.add(Resources::scene(&destination_sim));
    let mut total=live;total.add(resources);total.bounded()?;
    Ok(PreparedTransfer {id,source:source_id,destination:destination_id,source_mutation:source.mutation,
        destination_mutation:destination.mutation,source_epoch:source.epoch,destination_epoch:destination.epoch,
        time_nanos:source.time_nanos,source_sim,destination_sim,source_bodies,destination_bodies,
        source_epochs,destination_epochs,source_joints,destination_joints,resources,
        moved_bodies:count,moved_joints:moved_joints.len(),moved_contacts:contacts.len()})
}

fn dispatch(registry:&mut Registry,scene:i64,op:i32,ids:&[i64],values:&[f64])->Result<Vec<i64>,String> {
    match op {
        0 => {
            if ids.len()!=1 {return Err("body identity requires one ID".into());}
            let region=lookup(registry,scene)?;let handle=*region.bodies.get(&ids[0]).ok_or("unknown body")?;
            let (slot,generation)=handle.into_raw_parts();
            Ok(vec![ids[0],region.body_epochs[&ids[0]],slot as i64,generation as i64,region.epoch,region.time_nanos])
        },
        1 => {
            if ids.len()!=3||ids[0]<=0 {return Err("joint requires stable ID and two body IDs".into());} require(values,6)?;
            let a=vec(values,0)?;let b=vec(values,3)?;
            let region=registry.scenes.get_mut(&scene).ok_or("stale scene")?;
            if region.joints.contains_key(&ids[0]) || region.sim.impulse_joint_set.len()>=4096 {return Err("joint identity or capacity conflict".into());}
            let first=*region.bodies.get(&ids[1]).ok_or("unknown joint body")?;let second=*region.bodies.get(&ids[2]).ok_or("unknown joint body")?;
            if first==second{return Err("self joint".into());}
            let handle=region.sim.impulse_joint_set.insert(first,second,FixedJointBuilder::new().local_anchor1(a).local_anchor2(b).contacts_enabled(false),true);
            region.joints.insert(ids[0],handle);region.mutation+=1;Ok(vec![ids[0]])
        },
        2 => {
            if ids.len()!=1 {return Err("joint removal requires one ID".into());}
            let region=registry.scenes.get_mut(&scene).ok_or("stale scene")?;
            let handle=*region.joints.get(&ids[0]).ok_or("unknown joint")?;
            region.sim.impulse_joint_set.remove(handle,true).ok_or("stale joint")?;
            region.joints.remove(&ids[0]);region.mutation+=1;Ok(vec![])
        },
        3 => {
            if ids.len()!=1 {return Err("joint identity requires one ID".into());}
            let region=lookup(registry,scene)?;let handle=*region.joints.get(&ids[0]).ok_or("unknown joint")?;
            let joint=region.sim.impulse_joint_set.get(handle).ok_or("stale joint")?;
            let first=region.bodies.iter().find(|(_,h)|**h==joint.body1).ok_or("unknown first joint body")?.0;
            let second=region.bodies.iter().find(|(_,h)|**h==joint.body2).ok_or("unknown second joint body")?.0;
            let (slot,generation)=handle.into_raw_parts();Ok(vec![ids[0],*first,*second,slot as i64,generation as i64])
        },
        4 => {
            if registry.transfer.is_some(){return Err("transfer staging capacity exhausted".into());}
            let id=registry.next_transfer.checked_add(1).ok_or("transition identity exhausted")?;
            let prepared=prepare(registry,scene,ids,values,id)?;
            let receipt=vec![id,prepared.source_mutation,prepared.destination_mutation,prepared.time_nanos,
                prepared.moved_bodies as i64,prepared.moved_joints as i64,prepared.moved_contacts as i64];
            registry.transfer=Some(prepared);registry.next_transfer=id;Ok(receipt)
        },
        5 => {
            if ids.len()!=1{return Err("commit requires transition ID".into());}
            let transfer=registry.transfer.as_ref().ok_or("unknown transfer")?;
            if transfer.id!=ids[0]||transfer.source!=scene{return Err("stale transition identity".into());}
            let source=lookup(registry,transfer.source)?;let destination=lookup(registry,transfer.destination)?;
            if source.mutation!=transfer.source_mutation || destination.mutation!=transfer.destination_mutation
                || source.epoch!=transfer.source_epoch || destination.epoch!=transfer.destination_epoch
                || source.time_nanos!=transfer.time_nanos || destination.time_nanos!=transfer.time_nanos {
                return Err("transfer preparation became stale; abort and retry".into());
            }
            let transfer=registry.transfer.take().unwrap();
            let source=registry.scenes.get_mut(&transfer.source).unwrap();
            source.sim=transfer.source_sim;source.bodies=transfer.source_bodies;source.body_epochs=transfer.source_epochs;
            source.joints=transfer.source_joints;source.mutation+=1;
            let destination=registry.scenes.get_mut(&transfer.destination).unwrap();
            destination.sim=transfer.destination_sim;destination.bodies=transfer.destination_bodies;destination.body_epochs=transfer.destination_epochs;
            destination.joints=transfer.destination_joints;destination.mutation+=1;
            Ok(vec![transfer.id,transfer.time_nanos,transfer.moved_bodies as i64])
        },
        6 => {
            if ids.len()!=1{return Err("abort requires transition ID".into());}
            if !registry.transfer.as_ref().is_some_and(|t|t.id==ids[0]&&t.source==scene){return Err("stale transition identity".into());}
            registry.transfer=None;Ok(vec![])
        },
        7 => Ok(registry.transfer.as_ref().map(|t|t.resources.values()).unwrap_or_else(||Resources::default().values())),
        8 => {
            if ids.len()!=1{return Err("section query requires one lease".into());}
            let region=lookup(registry,scene)?;let section=region.sections.get(&ids[0]).ok_or("unknown section lease")?;
            Ok(vec![ids[0],section.revision,section.fingerprint as i64,if section.resident{1}else{0}])
        },
        9 | 10 => {
            if ids.len()!=5 {return Err("motion update requires complete body/frame lease".into());}
            require(values,if op==9{13}else{7})?;let target=pose(values)?;
            let linear=if op==9{vec(values,7)?}else{Vec3::ZERO};let angular=if op==9{vec(values,10)?}else{Vec3::ZERO};
            let region=registry.scenes.get_mut(&scene).ok_or("stale scene")?;
            let handle=body_lease(region,ids)?;
            let body=&region.sim.rigid_body_set[handle];
            if op==10&&!body.is_kinematic(){return Err("queued pose requires kinematic body".into());}
            for collider in body.colliders() {
                let c=&region.sim.collider_set[*collider];let aabb=c.shape().compute_aabb(&(target*c.position_wrt_parent().unwrap()));
                bounded(aabb.mins)?;bounded(aabb.maxs)?;
            }
            let body=&mut region.sim.rigid_body_set[handle];
            if op==10 {body.set_next_kinematic_position(target);}
            else {body.set_position(target,true);body.set_linvel(linear,true);body.set_angvel(angular,true);}
            region.mutation+=1;Ok(vec![])
        },
        11 => {require(values,0)?;Ok(vec![3])},
        12 => {
            if ids.len()!=5{return Err("body properties require complete body/frame lease".into());}require(values,12)?;
            if values.iter().any(|v|!v.is_finite())||values[0]<=0.||values[0]>1e9
                ||values[1]<0.||values[1]>100.||values[2]<0.||values[2]>100.
                ||!(values[3]==0.||values[3]==1.)||values[4]<0.||values[4]>16.
                ||values[5]<0.||values[5]>63.||values[5].fract()!=0.
                ||values[6]<0.||values[6]>10.||values[7]<0.||values[7]>1.||values[8]<0.||values[8]>1.
                ||values[9]<0.||values[10]<0.||values[11]<0.||values[11]>60. {
                return Err("invalid bounded body properties".into());
            }
            let region=registry.scenes.get_mut(&scene).ok_or("stale scene")?;
            let handle=body_lease(region,ids)?;
            let body=&region.sim.rigid_body_set[handle];
            if body.colliders().len()!=1{return Err("property update supports one box collider".into());}
            let collider=body.colliders()[0];
            let c=&mut region.sim.collider_set[collider];c.set_mass(values[0]);c.set_friction(values[6]);c.set_restitution(values[7]);c.set_contact_skin(values[8]);
            let body=&mut region.sim.rigid_body_set[handle];body.set_linear_damping(values[1]);body.set_angular_damping(values[2]);body.enable_ccd(values[3]!=0.);
            body.set_soft_ccd_prediction(values[4]);body.set_locked_axes(LockedAxes::from_bits(values[5] as u8).unwrap(),true);
            body.recompute_mass_properties_from_colliders(&region.sim.collider_set);
            let activation=body.activation_mut();activation.normalized_linear_threshold=values[9];activation.angular_threshold=values[10];activation.time_until_sleep=values[11];
            region.mutation+=1;Ok(vec![])
        },
        13 => {
            if !ids.is_empty(){return Err("scene clock query takes no identities".into());}require(values,0)?;
            let region=lookup(registry,scene)?;
            Ok(vec![scene,region.epoch,region.time_nanos,region.mutation])
        },
        14 => {
            if ids.len()!=5{return Err("clock initialization requires complete expected scene clock and target time".into());}require(values,0)?;
            if registry.transfer.is_some(){return Err("cannot initialize a scene clock while transfer staging exists".into());}
            if character::owns_scene(&registry.characters,scene){return Err("character ownership prevents empty-scene clock initialization".into());}
            let region=registry.scenes.get_mut(&scene).ok_or("stale scene")?;
            if region.failed_range || scene!=ids[0] || region.epoch!=ids[1] || region.time_nanos!=ids[2] || region.mutation!=ids[3] {
                return Err("stale scene clock initialization lease".into());
            }
            if ids[4]<0 || ids[4]<region.time_nanos {return Err("simulation time cannot move backwards".into());}
            if !region.bodies.is_empty() || !region.joints.is_empty() || region.sim.impulse_joint_set.len()!=0
                || region.sim.multibody_joint_set.iter().next().is_some()
                || region.sim.rigid_body_set.iter().any(|(_,body)|!body.is_fixed()) {
                return Err("only an actor-free scene can adopt the shared simulation clock".into());
            }
            let next_mutation=region.mutation.checked_add(1).ok_or("scene mutation exhausted")?;
            region.sim.initialize_static_indexes();
            region.time_nanos=ids[4];region.mutation=next_mutation;
            Ok(vec![scene,region.epoch,region.time_nanos,region.mutation])
        },
        20..=27 => character::dispatch(registry,scene,op,ids,values),
        30..=33 => terrain_batch::dispatch(registry,scene,op,ids,values),
        _=>Err("unknown typed foundation operation".into()),
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_dev_planetarysable_world_physics_FoundationNative_exchange<'a>(
    mut env:JNIEnv<'a>,_class:JClass<'a>,scene:jlong,op:jint,identities:JLongArray<'a>,values:JDoubleArray<'a>)->jlongArray {
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||->Result<Vec<i64>,String>{
        let n=env.get_array_length(&identities).map_err(|e|e.to_string())?;
        let count=env.get_array_length(&values).map_err(|e|e.to_string())?;
        if n>20485||count>64{return Err("oversized typed foundation payload".into());}
        let mut ids=vec![0;n as usize];let mut v=vec![0.;count as usize];
        env.get_long_array_region(&identities,0,&mut ids).map_err(|e|e.to_string())?;
        env.get_double_array_region(&values,0,&mut v).map_err(|e|e.to_string())?;
        let mut registry=REGISTRY.get_or_init(||Mutex::new(Registry::default())).lock().map_err(|_|"native registry poisoned")?;
        dispatch(&mut registry,scene,op,&ids,&v)
    }));
    let output=match result {
        Ok(Ok(result))=>result,
        Ok(Err(error))=>{let _=env.throw_new("java/lang/IllegalArgumentException",error);return std::ptr::null_mut();},
        Err(_)=>{let _=env.throw_new("java/lang/IllegalStateException","foundation transfer panic");return std::ptr::null_mut();},
    };
    match env.new_long_array(output.len() as i32) {
        Ok(array)=>{if env.set_long_array_region(&array,0,&output).is_err(){return std::ptr::null_mut();}array.into_raw()},
        Err(_)=>std::ptr::null_mut(),
    }
}
