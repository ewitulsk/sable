//! Immutable input-bound post-motion forces; no caller-authored final velocity.
use super::*;
use rapier3d_f64::parry::shape::Shape;
#[derive(Clone)]
pub(super) struct Rules { ids:Vec<i64>, values:Vec<i64>, acceleration:Vec3, up:Vec3, flight:i64, component:f64 }
fn vector(words:&[i64],at:usize)->Vec3 {Vec3::new(f64::from_bits(words[at] as u64),f64::from_bits(words[at+1] as u64),f64::from_bits(words[at+2] as u64))}
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
    registry.controlled.actors.get_mut(&ids[0]).unwrap().input.as_mut().unwrap().post_rules=Some(Rules {ids:ids.to_vec(),values:encoded,acceleration,up,flight:ids[10],component:values[6]});
    Ok(vec![ids[7],ids[8],ids[9],ids[10]])
}
#[derive(Clone,Copy)]
struct Support { normal:Vec3,carrier:Vec3,point:Vec3,body:i64,epoch:i64 }
fn primitive_support(region:&Region,own_shape:&dyn Shape,own_pose:&Pose,shape:&dyn Shape,pose:&Pose,
                     parent:Option<RigidBodyHandle>,up:Vec3,velocity:Vec3,best:&mut Option<Support>,work:&mut usize)->Result<(),String>{
    *work+=1;if *work>16384{return Err("post-motion endpoint primitive cap".into());}
    if let Some(compound)=shape.as_compound(){
        for (local,part) in compound.shapes(){primitive_support(region,own_shape,own_pose,part.as_ref(),&(*pose * *local),parent,up,velocity,best,work)?;}
        return Ok(());
    }
    if shape.as_cuboid().is_none(){return Err("unsupported post-motion endpoint primitive".into());}
    let Some(contact)=rapier3d_f64::parry::query::contact(own_pose,own_shape,pose,shape,0.003).map_err(|_|"unsupported endpoint contact query")? else{return Ok(());};
    if contact.dist < -0.005{return Err("post-motion endpoint penetration exceeds physical bound".into());}
    let normal=contact.normal2;
    if !normal.is_finite()||(normal.length()-1.).abs()>1e-6{return Err("invalid endpoint normal".into());}
    if normal.dot(up)<0.5{return Ok(());}
    let carrier=parent.map_or(Vec3::ZERO,|handle|region.sim.rigid_body_set[handle].velocity_at_point(contact.point2));
    if !carrier.is_finite()||(velocity-carrier).dot(normal)>0.02{return Ok(());}
    let body=parent.and_then(|h|region.bodies.iter().find(|(_,value)|**value==h).map(|(id,_)|*id)).unwrap_or(0);
    let support=Support {normal,carrier,point:contact.point2,body,epoch:if body==0{0}else{region.body_epochs[&body]}};
    if best.as_ref().map_or(true,|current|normal.dot(up)>current.normal.dot(up)){*best=Some(support);}Ok(())
}
fn endpoint(region:&Region,a:&Actor,up:Vec3,velocity:Vec3)->Result<Option<Support>,String>{
    let body=actor_body(region,a)?;if body.colliders().len()!=1{return Err("endpoint actor shape count changed".into());}
    if region.sim.collider_set.len()>8192{return Err("post-motion endpoint collider cap".into());}
    let own=body.colliders()[0];let collider=&region.sim.collider_set[own];
    let pose=*body.position()*collider.position_wrt_parent().ok_or("endpoint actor parent missing")?;
    let bounds=collider.shape().compute_aabb(&pose).loosened(0.003);let mut candidates=0;let mut work=0;let mut best=None;
    for (handle,other) in region.sim.collider_set.iter(){
        if handle==own||!other.is_enabled()||other.is_sensor()||!collider.collision_groups().test(other.collision_groups())||!collider.solver_groups().test(other.solver_groups()){continue;}
        let actual=other.parent().map_or(*other.position(),|h|*region.sim.rigid_body_set[h].position()*other.position_wrt_parent().unwrap());
        if !bounds.intersects(&other.shape().compute_aabb(&actual)){continue;}
        candidates+=1;if candidates>64{return Err("post-motion endpoint candidate cap".into());}
        primitive_support(region,collider.shape(),&pose,other.shape(),&actual,other.parent(),up,velocity,&mut best,&mut work)?;
    }Ok(best)
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
    let support=endpoint(region,a,rules.up,before)?;
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
        terminal_projection(dragged,axis*dragged.dot(axis),axis*(rules.component*0.6),&input.events)?.0
    };
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
