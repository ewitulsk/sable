//! Immutable input-bound post-motion forces; no caller-authored final velocity.
use super::*;
use rapier3d_f64::parry::shape::Shape;
#[derive(Clone)]
pub(super) struct Rules { ids:Vec<i64>, values:Vec<i64>, acceleration:Vec3, up:Vec3, flight:i64, component:f64, pub(super) incoming:Vec3 }
fn vector(words:&[i64],at:usize)->Vec3 {Vec3::new(f64::from_bits(words[at] as u64),f64::from_bits(words[at+1] as u64),f64::from_bits(words[at+2] as u64))}
/// ITEM gravity is already part of its admitted drive. Only endpoint drag belongs here.
pub(super) fn prepare_item_drag(registry:&mut Registry,scene:i64,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if ids.len()!=10{return Err("item drag requires exact input".into());}require(values,3)?;
    let up=vec(values,0)?;if (up.length()-1.).abs()>1e-6{return Err("item drag needs finite unit gravity up".into());}
    let a=actor(registry,scene,ids)?;let region=transfer::lookup(registry,scene)?;let input=a.input.as_ref().ok_or("item drag input absent")?;
    if a.kind!=2||input.started||a.result.is_some()||ids[7..10]!=[input.sequence,input.start,input.end]||input.start!=region.time_nanos||input.end-input.start!=50_000_000 {
        return Err("item drag requires unstarted ITEM50ms input".into());
    }
    if input.item_drag_up.is_some_and(|old|old!=up){return Err("item drag replay changed captured up".into());}
    registry.controlled.actors.get_mut(&ids[0]).unwrap().input.as_mut().unwrap().item_drag_up=Some(up);
    let mut receipt=ids[7..10].to_vec();receipt.extend(vector_bits(up));Ok(receipt)
}
pub(super) fn item_drag(region:&Region,a:&Actor,up:Vec3,velocity:Vec3)->Result<Vec3,String>{
    let (support,forces,_)=endpoint(region,a,up,velocity,false)?;
    let carrier=forces.map_or(Vec3::ZERO,|s|s.carrier);
    let axis=forces.map_or(up,|s|s.normal);
    let relative=velocity-carrier;let normal=axis*relative.dot(axis);
    let after=carrier+(relative-normal)*(if support.is_some(){0.588}else{0.98})+normal*0.98;
    if !after.is_finite()||after.length()>MAX_SPEED{return Err("item endpoint drag escaped admitted speed".into());}
    Ok(after)
}
pub(super) fn prepare(registry:&mut Registry,scene:i64,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if ids.len()!=11||!(0..=2).contains(&ids[10]) {return Err("post-motion rules need exact input and flight mode".into());}require(values,11)?;
    let a=actor(registry,scene,ids)?;let region=transfer::lookup(registry,scene)?;let input=a.input.as_ref().ok_or("post-motion input absent")?;
    if a.kind!=1||input.started||a.result.is_some()||ids[7..10]!=[input.sequence,input.start,input.end]||input.start!=region.time_nanos||input.end-input.start!=50_000_000 {
        return Err("post-motion rules require unstarted complete PLAYER50ms input".into());
    }
    let encoded:Vec<_>=values.iter().copied().map(bits).collect();
    if let Some(rules)=&input.post_rules {if rules.ids==ids&&rules.values==encoded{return Ok(vec![ids[7],ids[8],ids[9],ids[10]]);}return Err("post-motion rule replay differs".into());}
    let acceleration=Vec3::new(values[0],values[1],values[2]);let up=vec(values,3)?;
    let rotation=rapier3d_f64::glamx::DQuat::from_xyzw(values[7],values[8],values[9],values[10]);
    let actual=actor_body(region,a)?.position().rotation;
    if !acceleration.is_finite()||acceleration.length()>6400.||(up.length()-1.).abs()>1e-6||values[6].abs()>MAX_SPEED
        ||(rotation.length_squared()-1.).abs()>1e-6||(1.-rotation.dot(actual).abs()).abs()>1e-10||(actual*Vec3::Y-up).length()>1e-8 {
        return Err("post-motion captured field/orientation/flight bound or CAS mismatch".into());
    }
    // Preserve the actual native sample before the first controller drive replaces it.
    // A Java collision-clipped leg must not become the incoming landing speed.
    let incoming=actor_body(region,a)?.linvel();
    registry.controlled.actors.get_mut(&ids[0]).unwrap().input.as_mut().unwrap().post_rules=Some(Rules {ids:ids.to_vec(),values:encoded,acceleration,up,flight:ids[10],component:values[6],incoming});
    Ok(vec![ids[7],ids[8],ids[9],ids[10]])
}
#[derive(Clone,Copy)]
struct Support { normal:Vec3,carrier:Vec3,point:Vec3,body:i64,epoch:i64 }
#[derive(Clone,Copy)]
struct SupportForces { normal:Vec3,carrier:Vec3 }
#[derive(Clone,Copy)]
struct EndpointConstraint { normal:Vec3,carrier:Vec3,collider:ColliderHandle }

fn vector_order(left:Vec3,right:Vec3)->std::cmp::Ordering {
    left.x.total_cmp(&right.x).then(left.y.total_cmp(&right.y)).then(left.z.total_cmp(&right.z))
}
fn support_order(left:&Support,right:&Support)->std::cmp::Ordering {
    vector_order(left.normal,right.normal).then(left.body.cmp(&right.body)).then(left.epoch.cmp(&right.epoch))
        .then(vector_order(left.point,right.point)).then(vector_order(left.carrier,right.carrier))
}
/// Resolve simultaneous resting faces into one force frame without inventing a contact
/// for publication. The strongest real face retains provenance and its exact carrier.
/// Fixed faces may aggregate with other fixed faces; moving faces may aggregate only
/// within the same body ownership epoch. Equal-weight unique normals prevent repeated
/// manifold points or broadphase enumeration order from biasing a voxel-corner axis.
fn resolve_supports(mut supports:Vec<Support>,up:Vec3)->Result<Option<(Support,SupportForces)>,String>{
    if supports.is_empty(){return Ok(None);}
    supports.sort_by(support_order);
    let primary=*supports.iter().max_by(|left,right|left.normal.dot(up).total_cmp(&right.normal.dot(up)).then(support_order(left,right))).unwrap();
    let compatible=|candidate:&Support| (primary.body==0&&candidate.body==0)
        ||(primary.body==candidate.body&&primary.epoch==candidate.epoch);
    let mut unique:Vec<Vec3>=Vec::new();
    for support in supports {
        if !compatible(&support){continue;}
        let length=support.normal.length();
        if !length.is_finite()||length<1e-12{return Err("invalid endpoint support normal".into());}
        let normal=support.normal/length;
        if unique.iter().all(|existing|existing.dot(normal)<=1.-1e-10){unique.push(normal);}
    }
    let normal=unique.into_iter().fold(Vec3::ZERO,|sum,value|sum+value);let length=normal.length();
    if !normal.is_finite()||length<1e-12{return Err("invalid aggregate endpoint support frame".into());}
    Ok(Some((primary,SupportForces {normal:normal/length,carrier:primary.carrier})))
}
fn resting_support(normal:Vec3,up:Vec3,witness:Vec3,carrier:Vec3)->bool {
    normal.dot(up)>=0.5&&(witness-carrier).dot(normal)<=0.02
}
/// PLAYER grounding follows the final admitted path leg, not a retained solver response
/// or the later terminal-controller velocity correction. Completion proves every segment
/// executed through its exact end boundary before this witness can be consumed.
fn player_support_witness(input:&Input)->Result<Vec3,String> {
    let final_segment=input.segments.last().ok_or("player support requires a final admitted motor leg")?;
    if !input.started||input.active_segment+1!=input.segments.len()||final_segment.end!=input.end {
        return Err("player support requires the executed final motor leg".into());
    }
    Ok(final_segment.velocity)
}
// Minimize change to the requested controller impulse, never project actual velocity.
// The feasible origin guarantees that an existing solver response can remain untouched.
// At most three independent active planes define the Euclidean projection in 3D.
fn endpoint_motor(actual:Vec3,requested:Vec3,events:&[[i64;16]],contacts:&[EndpointConstraint])->Result<Vec3,String>{
    endpoint_motor_bounded(actual,requested,events,contacts,&[])
}
fn endpoint_motor_bounded(actual:Vec3,requested:Vec3,events:&[[i64;16]],contacts:&[EndpointConstraint],bounds:&[(Vec3,f64)])->Result<Vec3,String>{
    if contacts.len()>64{return Err("endpoint motor constraint cap".into());}
    let project=|value:Vec3| { let scale=value.length().max(1.); terminal_projection(Vec3::ZERO,Vec3::ZERO,value/scale,events).map(|v|v.0*scale) };
    let wanted=project(requested)?;
    let basis=[project(Vec3::X)?,project(Vec3::Y)?,project(Vec3::Z)?];
    let mut planes:Vec<(Vec3,f64)>=Vec::with_capacity(contacts.len());
    for contact in contacts {
        let n=contact.normal;
        let q=basis[0]*n.x+basis[1]*n.y+basis[2]*n.z;
        // Outward motion may decelerate to rest, but the new motor may neither create
        // inward relative motion nor worsen any inward solver response already present.
        let lower=(-(actual-contact.carrier).dot(n)).min(0.);
        if !q.is_finite()||!lower.is_finite(){return Err("invalid endpoint motor plane".into());}
        if let Some((_,old))=planes.iter_mut().find(|(axis,_)|*axis==q){*old=f64::max(*old,lower);}else{planes.push((q,lower));}
    }
    for (n,lower) in bounds {
        let q=basis[0]*n.x+basis[1]*n.y+basis[2]*n.z;
        if !q.is_finite()||!lower.is_finite(){return Err("invalid endpoint motor cancellation bound".into());}
        if let Some((_,old))=planes.iter_mut().find(|(axis,_)|*axis==q){*old=f64::max(*old,*lower);}else{planes.push((q,*lower));}
    }
    if planes.len()>64{return Err("endpoint motor total constraint cap".into());}
    let feasible=|value:Vec3| value.is_finite()&&planes.iter().all(|(q,b)|q.dot(value)>=*b-1e-8);
    if feasible(wanted){return Ok(wanted);}
    let mut work=0;
    let mut candidate=|active:&[usize]|->Result<Option<Vec3>,String>{
        work+=1;if work>43744{return Err("endpoint motor active-set cap".into());}
        let count=active.len();let mut rows=[[0.;4];3];
        for i in 0..count {
            let (q,b)=planes[active[i]];
            for j in 0..count{rows[i][j]=q.dot(planes[active[j]].0);}
            rows[i][count]=b-q.dot(wanted);
        }
        let scale=(0..count).map(|i|rows[i][i]).fold(0.,f64::max);
        if scale<1e-24{return Ok(None);}
        for col in 0..count {
            let pivot=(col..count).max_by(|a,b|rows[*a][col].abs().total_cmp(&rows[*b][col].abs())).unwrap();
            if rows[pivot][col].abs()<=scale*1e-12{return Ok(None);}
            rows.swap(pivot,col);let divisor=rows[col][col];
            for j in col..=count{rows[col][j]/=divisor;}
            let pivot_row=rows[col];for i in 0..count{if i!=col{let factor=rows[i][col];for j in col..=count{rows[i][j]-=factor*pivot_row[j];}}}
        }
        let mut result=wanted;
        for i in 0..count{let lambda=rows[i][count];if !lambda.is_finite()||lambda<0.{return Ok(None);}result+=planes[active[i]].0*lambda;}
        if !feasible(result)||active.iter().any(|i|(planes[*i].0.dot(result)-planes[*i].1).abs()>1e-8){return Ok(None);}
        // The positive multipliers, active equalities and all other halfspaces are the
        // complete KKT certificate. Reject any loss of the actual impulse nullspace.
        if project(result)?.distance(result)>1e-8{return Err("endpoint motor lost contact nullspace".into());}
        Ok(Some(result))
    };
    for i in 0..planes.len(){if let Some(value)=candidate(&[i])?{return Ok(value);}}
    for i in 0..planes.len(){for j in i+1..planes.len(){if let Some(value)=candidate(&[i,j])?{return Ok(value);}}}
    for i in 0..planes.len(){for j in i+1..planes.len(){for k in j+1..planes.len(){if let Some(value)=candidate(&[i,j,k])?{return Ok(value);}}}}
    Err("endpoint motor projection has no numerically verified solution".into())
}
fn primitive_support(region:&Region,own_shape:&dyn Shape,own_pose:&Pose,shape:&dyn Shape,pose:&Pose,
                     parent:Option<RigidBodyHandle>,collider:ColliderHandle,up:Vec3,velocity:Vec3,reach:f64,supports:&mut Vec<Support>,constraints:&mut Vec<EndpointConstraint>,collect:bool,work:&mut usize)->Result<(),String>{
    *work+=1;if *work>16384{return Err("post-motion endpoint primitive cap".into());}
    if let Some(compound)=shape.as_compound(){
        for (local,part) in compound.shapes(){primitive_support(region,own_shape,own_pose,part.as_ref(),&(*pose * *local),parent,collider,up,velocity,reach,supports,constraints,collect,work)?;}
        return Ok(());
    }
    let own=own_shape.as_cuboid().ok_or("unsupported endpoint actor primitive")?;
    let other=shape.as_cuboid().ok_or("unsupported post-motion endpoint primitive")?;
    // Use the same cuboid manifold clipping as the solver. The single-point contact
    // query can reject an exact face touch after selecting a far support vertex of
    // a large floor/ceiling and measuring its diagonal distance to the small actor.
    // This is a fresh actual-pose geometric query, never cached solver support.
    let mut manifold=rapier3d_f64::parry::query::ContactManifold::<(),()>::new();
    rapier3d_f64::parry::query::details::contact_manifold_cuboid_cuboid(&own_pose.inv_mul(pose),own,other,reach,&mut manifold);
    if manifold.points.is_empty(){return Ok(());}
    if manifold.points.len()>8{return Err("endpoint cuboid manifold point cap".into());}
    let normal=pose.rotation*manifold.local_n2;
    if !normal.is_finite()||(normal.length()-1.).abs()>1e-6{return Err("invalid endpoint normal".into());}
    for contact in &manifold.points {
        if !contact.dist.is_finite()||contact.dist < -0.005{return Err(format!("endpoint penetration: depth={:.9} ownHalf={:?} ownY={:.6} otherY={:.6} collider={:?}",contact.dist,own.half_extents,own_pose.translation.y,pose.translation.y,collider));}
        if contact.dist>reach{continue;}
        let point=*pose*contact.local_p2;
        let carrier=parent.map_or(Vec3::ZERO,|handle|region.sim.rigid_body_set[handle].velocity_at_point(point));
        if !point.is_finite()||!carrier.is_finite(){return Err("invalid endpoint carrier point velocity".into());}
        if collect {if constraints.len()>=64{return Err("post-motion endpoint constraint cap".into());} constraints.push(EndpointConstraint {normal,carrier,collider});}
        if !resting_support(normal,up,velocity,carrier){continue;}
        let body=parent.and_then(|h|region.bodies.iter().find(|(_,value)|**value==h).map(|(id,_)|*id)).unwrap_or(0);
        let support=Support {normal,carrier,point,body,epoch:if body==0{0}else{region.body_epochs[&body]}};
        supports.push(support);
    }
    Ok(())
}
fn endpoint(region:&Region,a:&Actor,up:Vec3,velocity:Vec3,collect:bool)->Result<(Option<Support>,Option<SupportForces>,Vec<EndpointConstraint>),String>{
    let body=actor_body(region,a)?;if body.colliders().len()!=1{return Err("endpoint actor shape count changed".into());}
    if region.sim.collider_set.len()>8192{return Err("post-motion endpoint collider cap".into());}
    let own=body.colliders()[0];let collider=&region.sim.collider_set[own];
    let pose=*body.position()*collider.position_wrt_parent().ok_or("endpoint actor parent missing")?;
    let bounds=collider.shape().compute_aabb(&pose);let mut candidates=0;let mut work=0;let mut supports=Vec::new();let mut constraints=Vec::new();
    for (handle,other) in region.sim.collider_set.iter(){
        if handle==own||!other.is_enabled()||other.is_sensor()||!collider.collision_groups().test(other.collision_groups())||!collider.solver_groups().test(other.solver_groups()){continue;}
        let actual=other.parent().map_or(*other.position(),|h|*region.sim.rigid_body_set[h].position()*other.position_wrt_parent().unwrap());
        // Compare the real two-collider skin distance, with one nanometre of
        // arithmetic tolerance shared by broadphase, manifold generation and filtering.
        let reach=0.003_f64.max(collider.contact_skin()+other.contact_skin())+1e-9;
        if !bounds.loosened(reach).intersects(&other.shape().compute_aabb(&actual)){continue;}
        candidates+=1;if candidates>64{return Err("post-motion endpoint candidate cap".into());}
        primitive_support(region,collider.shape(),&pose,other.shape(),&actual,other.parent(),handle,up,velocity,reach,&mut supports,&mut constraints,collect,&mut work)?;
    }
    let resolved=resolve_supports(supports,up)?;
    Ok((resolved.map(|value|value.0),resolved.map(|value|value.1),constraints))
}
/// A real endpoint can constrain motion without having generated a positive impulse
/// (for example a stationary skin-distance floor). Constrain only the next motor delta;
/// retained solver velocity and its positive-impulse nullspace remain untouched.
pub(super) fn controller_delta(region:&Region,a:&Actor,input:&Input,actual:Vec3,drive:Vec3,target:Vec3)->Result<(Vec3,usize),String>{
    let (projected,rank)=terminal_projection(actual,drive,target,&input.events)?;
    let Some(rules)=&input.post_rules else{return Ok((projected,rank));};
    let (_,_,constraints)=endpoint(region,a,rules.up,actual,true)?;
    let requested=target-drive;let mut retained=Vec::new();let mut bounds=Vec::new();
    for event in &input.events {
        let normal=vector(event,12);
        // A speculative solve can stop at contact while retaining inward motor speed.
        // Its outward impulse is real and remains in the physical ledger. Permit only
        // removal of the still-inward applied motor, toward relative rest, never beyond
        // it. An opposing impulse remains a hard nullspace plane and blocks this change.
        let cancellation=constraints.iter().filter(|c|{let (slot,generation)=c.collider.into_raw_parts();slot as i64==event[4]&&generation as i64==event[5]&&c.normal.dot(normal)>1.-1e-10})
            .map(|c|(-(actual-c.carrier).dot(normal)).min(-(drive-c.carrier).dot(normal)).min(requested.dot(normal)))
            .filter(|cap|*cap>1e-10).min_by(|a,b|a.total_cmp(b));
        if let Some(cap)=cancellation {
            bounds.push((normal,0.));bounds.push((-normal,-cap));
        }else{retained.push(*event);}
    }
    let delta=endpoint_motor_bounded(actual,requested,&retained,&constraints,&bounds)?;
    if !(actual+delta).is_finite()||(actual+delta).length()>MAX_SPEED{return Err("endpoint controller plus retained response exceeds speed envelope".into());}
    Ok((delta,rank))
}
pub(super) fn complete(registry:&mut Registry,scene:i64,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if ids.len()!=10{return Err("post-motion finalization requires exact input".into());}require(values,0)?;
    let a=actor(registry,scene,ids)?;let region=transfer::lookup(registry,scene)?;let input=a.input.as_ref().ok_or("post-motion input absent")?;
    let result=a.result.as_ref().ok_or("post-motion result not complete")?;
    if a.kind!=1||result[11..14]!=ids[7..10]||input.end!=region.time_nanos{return Err("post-motion result ownership mismatch".into());}
    if let Some(receipt)=&input.final_receipt {if vector(receipt,46)!=actor_body(region,a)?.linvel(){return Err("finalized native velocity changed".into());}return Ok(receipt.clone());}
    let rules=input.post_rules.as_ref().ok_or("post-motion rules were not captured before input")?;
    let terminal=input.terminal.as_ref().ok_or("terminal motor must precede post-motion forces")?;
    let before=actor_body(region,a)?.linvel();
    if vector(terminal,26)!=before||vector(result,21)!=before{return Err("post-motion actual terminal velocity CAS mismatch".into());}
    let support_witness=player_support_witness(input)?;
    let (support,forces,constraints)=endpoint(region,a,rules.up,support_witness,rules.flight!=0)?;
    let normal=forces.map_or(Vec3::ZERO,|s|s.normal);let carrier=forces.map_or(Vec3::ZERO,|s|s.carrier);
    let gravity=if rules.flight==0{
        let inward=rules.acceleration.dot(normal);
        (if support.is_some()&&inward<0.{normal*inward}else{rules.acceleration})*0.05
    }else{Vec3::ZERO};
    let accelerated=before+gravity;let axis=if support.is_some(){normal}else{rules.up};
    let relative=accelerated-carrier;let vertical=relative.dot(axis);
    let dragged=(relative-axis*vertical)*(if support.is_some(){0.546}else{0.91})+axis*(vertical*0.98)+carrier;
    let flight=if rules.flight==0{Vec3::ZERO}else{
        let axis=if rules.flight==1{Vec3::Y}else{rules.up};
        endpoint_motor(dragged,axis*(rules.component*0.6-dragged.dot(axis)),&input.events,&constraints)?
    };
    if input.events.iter().any(|event|flight.dot(vector(event,12)).abs()>1e-8){return Err("endpoint flight changed retained contact-normal response".into());}
    let after=dragged+flight;if !after.is_finite()||after.length()>MAX_SPEED{return Err("post-motion final velocity exceeds bound".into());}
    let next_mutation=region.mutation.checked_add(1).ok_or("post-motion mutation exhausted")?;
    let mut receipt=result[..14].to_vec();receipt.push(rules.flight);receipt.extend(vector_bits(rules.acceleration));receipt.extend(vector_bits(rules.up));receipt.push(bits(rules.component));
    receipt.extend(vector_bits(before));receipt.push(support.is_some() as i64);receipt.extend(vector_bits(support.map_or(Vec3::ZERO,|s|s.normal)));receipt.extend(vector_bits(support.map_or(Vec3::ZERO,|s|s.carrier)));
    receipt.extend([support.map_or(0,|s|s.body),support.map_or(0,|s|s.epoch)]);receipt.extend(vector_bits(support.map_or(Vec3::ZERO,|s|s.point)));
    receipt.extend(vector_bits(gravity));receipt.extend(vector_bits(dragged-accelerated));receipt.extend(vector_bits(flight));receipt.extend(vector_bits(after));
    let key=a.body;let region=registry.scenes.get_mut(&scene).unwrap();let body=&mut region.sim.rigid_body_set[region.bodies[&key]];
    body.set_linvel(after,true);body.planetary_refresh_motor_predictions();region.mutation=next_mutation;
    let a=registry.controlled.actors.get_mut(&ids[0]).unwrap();a.result.as_mut().unwrap()[21..24].copy_from_slice(&vector_bits(after));
    let input=a.input.as_mut().unwrap();input.applied_motor_delta+=after-before;input.final_receipt=Some(receipt.clone());Ok(receipt)
}

pub(super) fn lookup(registry:&Registry,scene:i64,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if ids.len()!=10{return Err("final receipt lookup requires exact input".into());}require(values,0)?;
    let a=actor(registry,scene,ids)?;let result=a.result.as_ref().ok_or("final receipt result absent")?;
    if result[11..14]!=ids[7..10]{return Err("final receipt lookup input mismatch".into());}
    Ok(a.input.as_ref().ok_or("final receipt input absent")?.final_receipt.clone().unwrap_or_default())
}

/// Separate controller-landing evidence: actual pre-drive native momentum resolved into
/// the confirmed endpoint material's contact frame. This is not a solver impulse or CCD
/// timestamp. No support, no walking mode, or no inward relative momentum means zero.
pub(super) fn landing(registry:&Registry,scene:i64,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if ids.len()!=10{return Err("controller landing requires exact final input".into());}require(values,0)?;
    let a=actor(registry,scene,ids)?;let result=a.result.as_ref().ok_or("controller landing result absent")?;
    if result[11..14]!=ids[7..10]{return Err("controller landing result mismatch".into());}
    let input=a.input.as_ref().ok_or("controller landing input absent")?;
    let rules=input.post_rules.as_ref().ok_or("controller landing rules absent")?;
    let final_result=input.final_receipt.as_ref().ok_or("controller landing endpoint not finalized")?;
    let speed=if rules.flight==0&&final_result[25]==1 {(-(rules.incoming-vector(final_result,29)).dot(vector(final_result,26))).max(0.)}else{0.};
    if !speed.is_finite(){return Err("controller landing relative speed invalid".into());}
    let mut out=result[..14].to_vec();out.extend(vector_bits(rules.incoming));out.push(bits(speed));Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn support(normal:Vec3,carrier:Vec3,body:i64)->Support {
        Support {normal,carrier,point:normal*body as f64,body,epoch:body}
    }
    fn corner(order:[usize;3],up:Vec3)->(Support,SupportForces) {
        let values=[
            support(Vec3::X,Vec3::ZERO,0),
            support(Vec3::Y,Vec3::ZERO,0),
            support(Vec3::Z,Vec3::ZERO,0),
        ];
        resolve_supports(order.into_iter().map(|index|values[index]).collect(),up).unwrap().unwrap()
    }
    fn player_input(final_velocity:Vec3)->Input {
        Input {sequence:1,start:0,end:50_000_000,velocity:final_velocity,started:true,initial_velocity:final_velocity,
            segments:vec![MotorSegment {end:50_000_000,velocity:final_velocity}],active_segment:0,applied_motor_delta:Vec3::ZERO,
            terminal:None,terminal_supported:true,post_rules:None,item_drag_up:None,final_receipt:None,
            events:Vec::new(),impacts:Vec::new(),landing_impacts:Vec::new()}
    }

    #[test]
    fn corner_force_frame_is_enumeration_order_invariant() {
        let up=Vec3::ONE.normalize();let expected=corner([0,1,2],up);
        for order in [[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]] {
            let actual=corner(order,up);
            assert_eq!(actual.0.normal,expected.0.normal);
            assert_eq!(actual.1.normal,expected.1.normal);
            assert_eq!(actual.1.carrier,expected.1.carrier);
        }
        assert!(expected.1.normal.distance(up)<1e-12);
        assert_eq!(expected.1.carrier,Vec3::ZERO);
    }

    #[test]
    fn repeated_points_on_one_face_do_not_bias_corner_normal() {
        let up=Vec3::ONE.normalize();
        let values=vec![
            support(Vec3::X,Vec3::ZERO,0),support(Vec3::X,Vec3::ZERO,0),support(Vec3::X,Vec3::ZERO,0),
            support(Vec3::Y,Vec3::ZERO,0),support(Vec3::Z,Vec3::ZERO,0),
        ];
        let (_,forces)=resolve_supports(values,up).unwrap().unwrap();
        assert!(forces.normal.distance(up)<1e-12);
    }

    #[test]
    fn near_coplanar_normals_deduplicate_after_normalization() {
        // Manifold normals are admitted within a small unit-length tolerance. Compare
        // directions, not their raw magnitudes, exactly as the Java mirror does.
        let almost_x=Vec3::new(0.9999996,0.000005,0.);
        let (_,forces)=resolve_supports(vec![support(Vec3::X,Vec3::ZERO,0),support(almost_x,Vec3::ZERO,0)],Vec3::X).unwrap().unwrap();
        assert!(forces.normal.distance(almost_x.normalize())<1e-12);

        // Just outside the 1e-10 directional threshold remains a second plane even
        // when its admitted raw magnitude would otherwise make the dot exceed one.
        let direction=Vec3::new(1.,0.000021,0.).normalize();let scaled=direction*1.0000005;
        let (_,forces)=resolve_supports(vec![support(Vec3::X,Vec3::ZERO,0),support(scaled,Vec3::ZERO,0)],Vec3::X).unwrap().unwrap();
        assert!(forces.normal.distance((Vec3::X+direction).normalize())<1e-12);
    }

    #[test]
    fn outward_solver_response_does_not_erase_resting_final_leg() {
        let input=player_input(Vec3::new(2.,-1.,0.));let actual=Vec3::new(2.,4.,0.);
        let witness=player_support_witness(&input).unwrap();
        assert_eq!(witness,input.segments[0].velocity);
        assert!(resting_support(Vec3::Y,Vec3::Y,witness,Vec3::ZERO));
        assert!(!resting_support(Vec3::Y,Vec3::Y,actual,Vec3::ZERO));
    }

    #[test]
    fn outward_final_leg_rejects_support_despite_inward_solver_velocity() {
        let input=player_input(Vec3::new(2.,1.,0.));let actual=Vec3::new(2.,-4.,0.);
        let witness=player_support_witness(&input).unwrap();
        assert!(!resting_support(Vec3::Y,Vec3::Y,witness,Vec3::ZERO));
        assert!(resting_support(Vec3::Y,Vec3::Y,actual,Vec3::ZERO));
    }

    #[test]
    fn corner_force_frame_changes_continuously_across_primary_tie() {
        let left=Vec3::new(0.999,1.,1.).normalize();let right=Vec3::new(1.001,1.,1.).normalize();
        let a=corner([0,1,2],left);let b=corner([0,1,2],right);
        assert_ne!(a.0.normal,b.0.normal,"the real strongest provenance face should cross the tie");
        assert!(a.1.normal.distance(Vec3::ONE.normalize())<1e-12);
        assert_eq!(a.1.normal,b.1.normal);
    }

    #[test]
    fn unrelated_moving_supports_do_not_mix_material_frames() {
        let up=Vec3::new(0.8,0.6,0.);let carrier=Vec3::ZERO;
        let supports=vec![support(Vec3::X,carrier,1),support(Vec3::Y,Vec3::ZERO,2)];
        let (primary,forces)=resolve_supports(supports,up).unwrap().unwrap();
        assert_eq!(primary.body,1);assert_eq!(forces.normal,Vec3::X);assert_eq!(forces.carrier,carrier);
    }

    #[test]
    fn one_moving_body_may_contribute_multiple_faces_without_averaging_carrier() {
        let up=Vec3::new(0.8,0.6,0.);let carrier=Vec3::new(3.,0.,0.);
        let mut x=support(Vec3::X,carrier,7);let mut y=support(Vec3::Y,Vec3::new(0.,4.,0.),7);x.epoch=9;y.epoch=9;
        let (primary,forces)=resolve_supports(vec![y,x],up).unwrap().unwrap();
        assert_eq!(primary.body,7);assert!(forces.normal.distance(Vec3::new(1.,1.,0.).normalize())<1e-12);
        assert_eq!(forces.carrier,carrier);
    }
}

