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
    let (support,_)=endpoint(region,a,up,velocity,false)?;
    let carrier=support.map_or(Vec3::ZERO,|s|s.carrier);
    let axis=support.map_or(up,|s|s.normal);
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
struct EndpointConstraint { normal:Vec3,carrier:Vec3 }
// Minimize change to the requested controller impulse, never project actual velocity.
// The feasible origin guarantees that an existing solver response can remain untouched.
// At most three independent active planes define the Euclidean projection in 3D.
fn endpoint_motor(actual:Vec3,requested:Vec3,events:&[[i64;16]],contacts:&[EndpointConstraint])->Result<Vec3,String>{
    if contacts.len()>64{return Err("endpoint motor constraint cap".into());}
    let project=|value:Vec3| { let scale=value.length().max(1.); terminal_projection(Vec3::ZERO,Vec3::ZERO,value/scale,events).map(|v|v.0*scale) };
    let wanted=project(requested)?;
    let basis=[project(Vec3::X)?,project(Vec3::Y)?,project(Vec3::Z)?];
    let mut planes=Vec::with_capacity(contacts.len());
    for contact in contacts {
        let n=contact.normal;
        let q=basis[0]*n.x+basis[1]*n.y+basis[2]*n.z;
        // Outward motion may decelerate to rest, but the new motor may neither create
        // inward relative motion nor worsen any inward solver response already present.
        let lower=(-(actual-contact.carrier).dot(n)).min(0.);
        if !q.is_finite()||!lower.is_finite(){return Err("invalid endpoint motor plane".into());}
        planes.push((q,lower));
    }
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
                     parent:Option<RigidBodyHandle>,up:Vec3,velocity:Vec3,reach:f64,best:&mut Option<Support>,constraints:&mut Vec<EndpointConstraint>,collect:bool,work:&mut usize)->Result<(),String>{
    *work+=1;if *work>16384{return Err("post-motion endpoint primitive cap".into());}
    if let Some(compound)=shape.as_compound(){
        for (local,part) in compound.shapes(){primitive_support(region,own_shape,own_pose,part.as_ref(),&(*pose * *local),parent,up,velocity,reach,best,constraints,collect,work)?;}
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
        if !contact.dist.is_finite()||contact.dist < -0.005{return Err("post-motion endpoint penetration exceeds physical bound".into());}
        if contact.dist>reach{continue;}
        let point=*pose*contact.local_p2;
        let carrier=parent.map_or(Vec3::ZERO,|handle|region.sim.rigid_body_set[handle].velocity_at_point(point));
        if !point.is_finite()||!carrier.is_finite(){return Err("invalid endpoint carrier point velocity".into());}
        if collect {if constraints.len()>=64{return Err("post-motion endpoint constraint cap".into());} constraints.push(EndpointConstraint {normal,carrier});}
        if normal.dot(up)<0.5||(velocity-carrier).dot(normal)>0.02{continue;}
        let body=parent.and_then(|h|region.bodies.iter().find(|(_,value)|**value==h).map(|(id,_)|*id)).unwrap_or(0);
        let support=Support {normal,carrier,point,body,epoch:if body==0{0}else{region.body_epochs[&body]}};
        if best.as_ref().map_or(true,|current|normal.dot(up)>current.normal.dot(up)){*best=Some(support);}
    }
    Ok(())
}
fn endpoint(region:&Region,a:&Actor,up:Vec3,velocity:Vec3,collect:bool)->Result<(Option<Support>,Vec<EndpointConstraint>),String>{
    let body=actor_body(region,a)?;if body.colliders().len()!=1{return Err("endpoint actor shape count changed".into());}
    if region.sim.collider_set.len()>8192{return Err("post-motion endpoint collider cap".into());}
    let own=body.colliders()[0];let collider=&region.sim.collider_set[own];
    let pose=*body.position()*collider.position_wrt_parent().ok_or("endpoint actor parent missing")?;
    let bounds=collider.shape().compute_aabb(&pose);let mut candidates=0;let mut work=0;let mut best=None;let mut constraints=Vec::new();
    for (handle,other) in region.sim.collider_set.iter(){
        if handle==own||!other.is_enabled()||other.is_sensor()||!collider.collision_groups().test(other.collision_groups())||!collider.solver_groups().test(other.solver_groups()){continue;}
        let actual=other.parent().map_or(*other.position(),|h|*region.sim.rigid_body_set[h].position()*other.position_wrt_parent().unwrap());
        // Compare the real two-collider skin distance, with one nanometre of
        // arithmetic tolerance shared by broadphase, manifold generation and filtering.
        let reach=0.003_f64.max(collider.contact_skin()+other.contact_skin())+1e-9;
        if !bounds.loosened(reach).intersects(&other.shape().compute_aabb(&actual)){continue;}
        candidates+=1;if candidates>64{return Err("post-motion endpoint candidate cap".into());}
        primitive_support(region,collider.shape(),&pose,other.shape(),&actual,other.parent(),up,velocity,reach,&mut best,&mut constraints,collect,&mut work)?;
    }Ok((best,constraints))
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
    let (support,constraints)=endpoint(region,a,rules.up,before,rules.flight!=0)?;
    let normal=support.map_or(Vec3::ZERO,|s|s.normal);let carrier=support.map_or(Vec3::ZERO,|s|s.carrier);
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
    receipt.extend(vector_bits(before));receipt.push(support.is_some() as i64);receipt.extend(vector_bits(normal));receipt.extend(vector_bits(carrier));
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

