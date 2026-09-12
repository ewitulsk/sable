//! Bounded staged intervals. Required terrain is derived from actual internal motion/query
//! observations, never from only the final speed or final pose. Unknown terrain is not air.
use super::*;
use std::collections::{BTreeSet, HashSet};
use rapier3d_f64::dynamics::PlanetarySweepTrace;
use rapier3d_f64::geometry::PlanetaryContactGuard;

const MAX_BODIES:usize=4096;
const MAX_CELLS:usize=4096;
const MAX_INTERVAL:i64=50_000_000;
const MAX_COMMANDS:usize=4096;
pub(super) struct Terminal { operation:i32, header:Vec<i64> }
pub(super) struct Preview {
    id:i64, scene:i64, frame:i64, start:i64, end:i64, mutation:i64,
    state:i64, commands:usize, candidate:Option<Box<Registry>>,
    traces:Vec<(i64,i64,RigidBodyHandle,PlanetarySweepTrace)>,
    coverage:Vec<i64>, missing:Vec<Vec3>,
    contact_guard:Option<PlanetaryContactGuard>,
    // One immutable full-interval path per ordinary position-kinematic body. Solver/motor
    // subdivisions sample this path; they must not consume its endpoint on the first step.
    kinematic_paths:Option<Vec<(RigidBodyHandle,Pose,Pose)>>,
}
pub(super) fn retains(registry:&Registry,scene:i64)->bool {
    registry.preview.as_ref().is_some_and(|p|p.scene==scene)
}
pub(super) fn candidate_sim(registry:&Registry)->Option<&Simulation> {
    let preview=registry.preview.as_ref()?;
    preview.candidate.as_ref()?.scenes.get(&preview.scene).map(|r|&r.sim)
}
impl Preview {
    fn header(&self)->Vec<i64> { vec![self.id,self.scene,self.frame,self.start,self.end,self.mutation,
        self.state,if self.state==0||self.state==3{-1}else{self.missing.len() as i64},self.traces.len() as i64] }
    fn open(&self)->Result<(),String> {
        if self.state!=0 {return Err("staged interval is sealed, committed or failed".into());}Ok(())
    }
    fn snapshot(&self)->Result<Vec<i64>,String> {
        let mut out=self.header();
        if self.state==1||self.state==2 {out.extend_from_slice(&self.coverage);}
        else {out.extend(self.trace_words()?);}
        for cell in &self.missing {out.extend([cell.x.to_bits() as i64,cell.y.to_bits() as i64,cell.z.to_bits() as i64]);}
        Ok(out)
    }
    fn trace_words(&self)->Result<Vec<i64>,String> {
        let mut rows=Vec::with_capacity(self.traces.len()*13);
        for (id,epoch,_,trace) in &self.traces {
            let b=trace.snapshot()?;
            rows.extend([*id,*epoch,b.mins.x.to_bits() as i64,b.mins.y.to_bits() as i64,b.mins.z.to_bits() as i64,
                b.maxs.x.to_bits() as i64,b.maxs.y.to_bits() as i64,b.maxs.z.to_bits() as i64,
                b.samples as i64,b.solver_segments as i64,b.ccd_predictions as i64,b.ccd_clamps as i64,b.broad_phase_queries as i64]);
        }Ok(rows)
    }
}

fn begin(registry:&mut Registry,scene:i64,ids:&[i64])->Result<Vec<i64>,String> {
    if ids.len()!=5 || ids[0]!=scene {return Err("preview requires exact scene clock and target time".into());}
    if registry.preview.is_some() || registry.transfer.is_some() || registry.mapped_receipt.is_some() {
        return Err("bounded native staging slot remains owned".into());
    }
    let source=transfer::lookup(registry,scene)?;
    if [source.epoch,source.time_nanos,source.mutation]!=ids[1..4] {
        return Err("stale staged interval clock".into());
    }
    if ids[4]<=source.time_nanos || ids[4].checked_sub(source.time_nanos).is_none_or(|d|d>MAX_INTERVAL) {
        return Err("staged interval duration exceeds 50ms".into());
    }
    if !source.streamed_terrain || !terrain_batch::idle(&source.terrain_batch)
        || character::owns_scene(&registry.characters,scene) {
        return Err("preview requires resident streamed terrain, no terrain receipt or legacy actors".into());
    }
    controlled::transfer_ready(registry,scene,scene)?;
    if source.bodies.len()>MAX_BODIES || source.sections.len()>MAX_SECTIONS
        || source.sim.multibody_joint_set.iter().next().is_some() {
        return Err("preview body/section capacity or untraced multibody topology".into());
    }
    let owned:HashSet<_>=source.bodies.values().copied().collect();
    if source.sim.rigid_body_set.iter().any(|(h,b)|!b.is_fixed()&&!owned.contains(&h)) {
        return Err("preview contains unowned moving body".into());
    }
    // Reserve memory for the clone before constructing it. Terrain parts are independently
    // budgeted below even though immutable SharedShapes may physically share allocation.
    transfer::preview_budget(registry,&source.sim)?;
    let mut sections=HashMap::with_capacity(source.sections.len());
    for (id,s) in &source.sections {
        sections.insert(*id,Section {revision:s.revision,body:s.body,collider:s.collider,
            _parts:PartBudget::reserve(s._parts.0)?,fingerprint:s.fingerprint,translation:s.translation,resident:s.resident});
    }
    let mut sim=source.sim.staged_clone();
    let mut traces=Vec::with_capacity(source.bodies.len());
    let mut bodies:Vec<_>=source.bodies.iter().map(|(id,h)|(*id,*h)).collect();bodies.sort_unstable_by_key(|v|v.0);
    for (id,h) in bodies {
        let body=&mut sim.rigid_body_set[h];
        if body.is_fixed() {continue;}
        // Pending mass changes must be settled before the trace uses COM coordinates.
        // Later changes invalidate the trace at both mass recomputation and solver capture.
        body.recompute_mass_properties_from_colliders(&sim.collider_set);
        let com=body.local_center_of_mass();
        let mut radius:f64=0.;
        for handle in body.colliders() {
            let collider=&sim.collider_set[*handle];
            let local=collider.shape().compute_aabb(collider.position_wrt_parent().ok_or("preview collider missing parent pose")?);
            radius=radius.max((local.mins-com).abs().max((local.maxs-com).abs()).length()+collider.contact_skin());
        }
        // The COM of an origin-interpolated kinematic rotation can arc away from its endpoint
        // chord by at most twice the local COM radius. Recorded native query bounds supplement
        // this full-orientation body envelope, including fat cached broadphase leaves.
        radius+=2.*com.length()+sim.parameters.prediction_distance()+0.02*sim.parameters.length_unit;
        if !radius.is_finite() {return Err("nonfinite staged body envelope".into());}
        traces.push((id,source.body_epochs[&id],h,body.planetary_begin_sweep(radius)));
    }
    let candidate=Region {terrain_batch:source.terrain_batch.clone(),sim,sections,bodies:source.bodies.clone(),
        epoch:source.epoch,failed_range:false,streamed_terrain:source.streamed_terrain,
        section_high_water:source.section_high_water,mutation:source.mutation,time_nanos:source.time_nanos,
        body_epochs:source.body_epochs.clone(),body_history:source.body_history.clone(),joints:source.joints.clone(),next_legacy_joint:source.next_legacy_joint};
    let mut staged=Registry::default();
    staged.controlled=controlled::snapshot_scene(&registry.controlled,scene);
    staged.scenes.insert(scene,candidate);
    let id=registry.next_preview.checked_add(1).ok_or("staged interval identity exhausted")?;
    let p=Preview {id,scene,frame:source.epoch,start:source.time_nanos,end:ids[4],mutation:source.mutation,
        state:0,commands:0,candidate:Some(Box::new(staged)),traces,coverage:vec![],missing:vec![],contact_guard:None,kinematic_paths:None};
    let result=p.header();registry.next_preview=id;registry.preview=Some(p);Ok(result)
}

fn command(p:&mut Preview,ids:&[i64],values:&[f64])->Result<Vec<i64>,String> {
    if ids.is_empty() || p.commands>=MAX_COMMANDS {return Err("staged command bound or absent operation".into());}
    let op=i32::try_from(ids[0]).map_err(|_|"staged command operation overflow")?;
    if matches!(op,3|13|42|44|58|72|74|76|77|79|83|84|85|86|90) {
        if p.state!=0&&p.state!=1 {return Err("candidate reads require open or sealed interval".into());}
    }else{p.open()?;}
    let ids=&ids[1..];
    if op==10&&p.kinematic_paths.is_some(){return Err("kinematic trajectory already belongs to the retained interval".into());}
    let candidate=p.candidate.as_mut().ok_or("staged scene absent")?;
    let result=match op {
        // Results remain owned across commit. An in-clone ACK would erase publication
        // obligations before the canonical owner had ever received the physical result.
        3|10|13|42|43|44|58|74..=79|81..=87=>transfer::dispatch(candidate,p.scene,op,ids,values),
        90=>controlled_pose::dispatch(candidate,p.scene,op,ids,values),
        91|92=>{
            if transfer::lookup(candidate,p.scene)?.time_nanos!=p.start {
                return Err("feet-anchored transition must precede candidate advancement".into());
            }
            controlled::transfer_ready(candidate,p.scene,p.scene)?;
            controlled_pose::dispatch(candidate,p.scene,op,ids,values)
        },
        70|71=>{
            if ids.len()!=5 {return Err("staged gravity/force requires complete body lease".into());}
            require(values,3)?;let field=vec(values,0)?;
            if controlled::owns_body(&candidate.controlled,p.scene,ids[0]) {
                return Err("controlled gravity belongs to the queued motion owner".into());
            }
            let r=candidate.scenes.get_mut(&p.scene).unwrap();
            let h=transfer::body_lease(r,ids)?;
            let next=r.mutation.checked_add(1).ok_or("staged mutation exhausted")?;
            let b=&mut r.sim.rigid_body_set[h];
            if !b.is_dynamic(){return Err("staged gravity/force requires dynamic body".into());}
            if op==70 {b.planetary_set_gravity(Some(field));}else{b.add_force(field,true);}
            r.mutation=next;Ok(vec![])
        },
        72=>{
            require(values,0)?;
            let r=transfer::lookup(candidate,p.scene)?;transfer::body_lease(r,ids)?;
            Ok(transfer::body_state(r,ids[0],ids[1])?.into_iter().map(|v|v.to_bits() as i64).collect())
        },
        _=>Err("command is not admitted inside staged interval".into()),
    };
    p.commands+=1;
    if result.is_err(){p.state=3;}result
}

fn seal(p:&mut Preview)->Result<Vec<i64>,String> {
    p.open()?;
    let r=transfer::lookup(p.candidate.as_ref().unwrap(),p.scene)?;
    if r.time_nanos!=p.end {return Err("staged interval must reach exact requested end before seal".into());}
    controlled::preview_complete(p.candidate.as_ref().unwrap(),p.scene)?;
    p.contact_guard.as_ref().ok_or("staged interval lacks contact allocation admission")?.snapshot()?;
    let origin=r.sections.values().filter(|s|s.resident).min_by_key(|s|(s.translation.x.to_bits(),s.translation.y.to_bits(),s.translation.z.to_bits()))
        .ok_or("preview requires resident terrain lattice")?.translation;
    let mut resident=HashSet::new();
    for section in r.sections.values().filter(|s|s.resident) {
        let rel=(section.translation-origin)/16.;
        if !rel.is_finite()||rel.x.fract()!=0.||rel.y.fract()!=0.||rel.z.fract()!=0. {
            return Err("resident terrain is not one exact 16m lattice".into());
        }
        if !resident.insert([rel.x as i64,rel.y as i64,rel.z as i64]) {
            return Err("duplicate resident terrain cell".into());
        }
    }
    let mut required=BTreeSet::new();
    for (_,_,_,trace) in &p.traces {
        let b=trace.snapshot()?;bounded(b.mins)?;bounded(b.maxs)?;
        let min=((b.mins-origin)/16.).floor();let max=((b.maxs-origin)/16.).floor();
        for x in min.x as i64..=max.x as i64 {for y in min.y as i64..=max.y as i64 {for z in min.z as i64..=max.z as i64 {
            required.insert([x,y,z]);
            if required.len()>MAX_CELLS {return Err("staged collision residency demand exceeds bounded region capacity".into());}
        }}}
    }
    p.coverage=p.trace_words()?;
    p.missing=required.into_iter().filter(|c|!resident.contains(c)).map(|c|origin+Vec3::new(c[0] as f64,c[1] as f64,c[2] as f64)*16.).collect();
    p.state=1;p.snapshot()
}

pub(super) fn dispatch(registry:&mut Registry,scene:i64,op:i32,ids:&[i64],values:&[f64])->Result<Vec<i64>,String> {
    if op==60 {require(values,0)?;if !ids.is_empty(){return Err("preview capability takes no identities".into());}return Ok(vec![1,MAX_BODIES as i64,MAX_CELLS as i64,MAX_INTERVAL]);}
    if op!=63 {require(values,0)?;}
    if op==61 {return begin(registry,scene,ids);}
    if ids.is_empty(){return Err("staged operation requires exact token".into());}
    if matches!(op,67|68|69)&&ids.len()==1 {
        if let Some(done)=registry.preview_terminal.get(&scene) {
            if done.header[0]==ids[0] {
                if op==67 {return Ok(done.header.clone());}
                if op==done.operation {return Ok(vec![ids[0]]);}
                return Err("staged terminal receipt has a different disposition".into());
            }
        }
    }
    if op==67&&ids.len()==5&&!retains(registry,scene) {
        let actual=transfer::lookup(registry,scene)?;
        if ids[..4]!=[scene,actual.epoch,actual.time_nanos,actual.mutation]||ids[4]<=actual.time_nanos
            ||ids[4].checked_sub(actual.time_nanos).is_none_or(|n|n>MAX_INTERVAL) {
            return Err("staged begin absence cannot be proved against changed source".into());
        }
        return Ok(vec![]);
    }
    let p=registry.preview.as_ref().ok_or("staged interval absent")?;
    // A lost begin response must be recoverable without guessing its newly allocated token.
    // The exact captured owner/clock/end tuple identifies the single retained preparation.
    if op==67&&ids.len()==5 {
        if [p.scene,p.frame,p.start,p.mutation,p.end]!=ids||p.scene!=scene {
            return Err("staged begin recovery descriptor mismatch".into());
        }
        return Ok(p.header());
    }
    if p.id!=ids[0]||p.scene!=scene {return Err("stale staged interval token/owner".into());}
    if op!=63&&op!=64&&ids.len()!=1 {return Err("staged operation has unexpected identities".into());}
    match op {
        62=>p.snapshot(),
        63=>{
            let p=registry.preview.as_mut().unwrap();
            let result=command(p,&ids[1..],values);
            // Include early validation failures, not only errors returned by the inner match.
            // A committed receipt is immutable even when a caller sends a stale command.
            if result.is_err()&&p.state!=2 {p.state=3;}
            result
        },
        64=>{
            if ids.len()!=2 {return Err("staged step requires token and nanoseconds".into());}
            let p=registry.preview.as_mut().unwrap();p.open()?;
            if p.commands>=MAX_COMMANDS{return Err("staged step command bound".into());}
            let c=p.candidate.as_ref().unwrap();
            let now=transfer::lookup(c,scene)?.time_nanos;
            if ids[1]<=0||now.checked_add(ids[1]).is_none_or(|t|t>p.end){return Err("staged step exceeds reserved interval".into());}
            if p.kinematic_paths.is_none(){
                let region=transfer::lookup(c,scene)?;
                let mut paths=Vec::new();
                for (_,_,handle,_) in &p.traces {
                    let body=&region.sim.rigid_body_set[*handle];
                    if body.body_type()==RigidBodyType::KinematicPositionBased{paths.push((*handle,*body.position(),*body.next_position()));}
                }
                p.kinematic_paths=Some(paths);
            }
            let fraction=(now+ids[1]-p.start) as f64/(p.end-p.start) as f64;
            let paths=p.kinematic_paths.as_ref().unwrap().clone();
            p.commands+=1;
            if let Some(guard)=&p.contact_guard {guard.snapshot()?;}
            let mut candidate=p.candidate.take().unwrap();
            let result=(||->Result<PlanetaryContactGuard,String>{
                // Recompute the ceiling while the registry lock excludes every competing
                // allocation. Original source and other retained clones still count.
                let guard=PlanetaryContactGuard::new(transfer::remaining_contact_budget(registry,None)?);
                let sim=&mut candidate.scenes.get_mut(&scene).unwrap().sim;
                // Terrain/actor collider retirement is deferred by Rapier until its next
                // pipeline maintenance. The allocation guard inspects contacts BEFORE that
                // pipeline step, so settle only exact retired generations in this clone.
                sim.flush_pending_removals();
                for (handle,start,end) in &paths {
                    let mut rotation=end.rotation;
                    if start.rotation.dot(rotation)<0.{rotation=-rotation;}
                    let target=if fraction==1. {*end}else{Pose {
                        translation:start.translation+(end.translation-start.translation)*fraction,
                        rotation:start.rotation.slerp(rotation,fraction),
                    }};
                    sim.rigid_body_set[*handle].set_next_kinematic_position(target);
                }
                sim.narrow_phase.planetary_clear_contact_guard();
                sim.narrow_phase.planetary_guard_contacts(&sim.collider_set,&guard)?;
                advance_region(&mut candidate,scene,ids[1])?;guard.snapshot()?;
                transfer::preview_budget(registry,&candidate.scenes[&scene].sim)?;
                Ok(guard)
            })();
            match result {
                Ok(guard)=>{let p=registry.preview.as_mut().unwrap();p.contact_guard=Some(guard);p.candidate=Some(candidate);},
                Err(e)=>{registry.preview.as_mut().unwrap().state=3;return Err(e);},
            }
            Ok(registry.preview.as_ref().unwrap().header())
        },
        65=>{if p.state==1{return p.snapshot();}seal(registry.preview.as_mut().unwrap())},
        66=>{
            if p.state==2{return Ok(p.header());}
            if p.state!=1||!p.missing.is_empty(){return Err("staged commit requires sealed complete collision terrain".into());}
            let r=transfer::lookup(registry,scene)?;
            if r.epoch!=p.frame||r.time_nanos!=p.start||r.mutation!=p.mutation {return Err("staged source clock changed before commit".into());}
            let c=p.candidate.as_ref().unwrap();let staged=transfer::lookup(c,scene)?;
            p.contact_guard.as_ref().ok_or("staged contact guard missing")?.snapshot()?;
            transfer::preview_budget(registry,&staged.sim)?;
            let p=registry.preview.as_mut().unwrap();
            let mut candidate=p.candidate.take().unwrap();
            let mut ready=candidate.scenes.remove(&scene).unwrap();
            for (_,_,h,_) in &p.traces {ready.sim.rigid_body_set[*h].planetary_end_sweep();}
            // PhysicsPipeline owns reusable scratch only. Retire solver-held Arc traces too.
            ready.sim.pipeline=PhysicsPipeline::new();
            ready.sim.narrow_phase.planetary_clear_contact_guard();
            registry.scenes.insert(scene,ready);
            controlled::publish_scene(&mut registry.controlled,scene,candidate.controlled);
            p.state=2;Ok(p.header())
        },
        67=>Ok(p.header()),
        68|69=>{
            if op==68&&p.state!=2{return Err("only committed staged receipt can be acknowledged".into());}
            if op==69&&p.state==2{return Err("committed interval cannot abort".into());}
            let mut header=p.header();header[6]=if op==68{4}else{5};
            // At most one terminal receipt per live scene. Java retains the outer interval
            // through this boundary, so a lost reply can retry its exact disposition.
            registry.preview_terminal.insert(scene,Terminal{operation:op,header});
            registry.preview=None;Ok(vec![ids[0]])
        },
        _=>Err("unknown staged interval operation".into()),
    }
}
