//! Feet-anchored PLAYER shape/orientation transition, admitted only in a staged preview.
//! The continuous path is checked conservatively; clear endpoints alone never authorize it.
use super::*;
use rapier3d_f64::parry::bounding_volume::BoundingVolume;
use rapier3d_f64::geometry::Aabb;
const MAX_INTERVALS:usize=64;
const MAX_COLLIDERS:usize=8192;
const MAX_QUERIES:usize=262144;
const EPS:f64=1e-9;
const PROJECTION_SKIN:f64=1e-5;
const MAX_CORRECTION:f64=0.35;

fn same_pose(a:&Pose,b:&Pose)->bool {
    (a.translation-b.translation).length()<=EPS && a.rotation.dot(b.rotation).abs()>=1.-1e-12
}
fn half(v:&[f64],at:usize)->Result<Vec3,String>{
    let h=vec(v,at)?;if h.min_element()<0.05||h.max_element()>2.{return Err("feet pose half extent outside controlled bounds".into());}Ok(h)
}
fn append_vec(out:&mut Vec<i64>,v:Vec3){out.extend([v.x.to_bits() as i64,v.y.to_bits() as i64,v.z.to_bits() as i64]);}
fn append_pose(out:&mut Vec<i64>,p:&Pose){append_vec(out,p.translation);out.extend([p.rotation.x.to_bits() as i64,p.rotation.y.to_bits() as i64,p.rotation.z.to_bits() as i64,p.rotation.w.to_bits() as i64]);}
struct Envelope {pose:Pose,half:Vec3,aabb:Aabb,translation:Option<(Pose,Pose,Vec3)>}
fn translation_envelope(start:Pose,end:Pose,half:Vec3)->Envelope {
    let actual_half=half;
    let local=start.rotation.inverse()*(end.translation-start.translation);
    let half=half+local.abs()*0.5;
    let pose=Pose {translation:(start.translation+end.translation)*0.5,rotation:start.rotation};
    let aabb=SharedShape::cuboid(half.x,half.y,half.z).compute_aabb(&pose);
    Envelope {pose,half,aabb,translation:Some((start,end,actual_half))}
}
/// A separating projection of the full translated box proves clearance continuously. This
/// refines the loose oriented hull on slanted floor approaches without endpoint-only tests.
fn translation_clear(start:&Pose,end:&Pose,half:Vec3,other:&Pose,other_half:Vec3,allowed:f64)->bool {
    let axes=[Vec3::X,Vec3::Y,Vec3::Z];
    let a=axes.map(|axis|start.rotation*axis);let b=axes.map(|axis|other.rotation*axis);
    let mut candidates=Vec::with_capacity(15);candidates.extend(a);candidates.extend(b);
    for x in a {for y in b {candidates.push(x.cross(y));}}
    for axis in candidates {
        let length=axis.length();if length<1e-12{continue;}let n=axis/length;
        let radius=half.dot((start.rotation.inverse()*n).abs())+other_half.dot((other.rotation.inverse()*n).abs());
        let p=(start.translation-other.translation).dot(n);let q=(end.translation-other.translation).dot(n);
        if p.min(q)>=radius-allowed-EPS||p.max(q)<=-radius+allowed+EPS{return true;}
    }
    false
}
fn charge(queries:&mut usize)->Result<(),String>{
    *queries=queries.checked_add(1).ok_or("feet transition query overflow")?;
    if *queries>MAX_QUERIES{return Err("feet transition primitive query capacity".into());}Ok(())
}

/// Minimum translation along up admitting a separating projection of these two boxes.
/// The feet plane is not the contact plane on a tilted stance. Use actual primitive axes;
/// this only proposes a lift, and all three continuous route segments are checked below.
fn clearance_lift(path:&Envelope,other:&Pose,other_half:Vec3,up:Vec3,allowed:f64)->f64 {
    if translation_clear(&path.pose,&path.pose,path.half,other,other_half,allowed){return 0.;}
    let basis=[Vec3::X,Vec3::Y,Vec3::Z];
    let a=basis.map(|e|path.pose.rotation*e);let b=basis.map(|e|other.rotation*e);
    let mut axes=Vec::with_capacity(15);axes.extend(a);axes.extend(b);
    for x in a {for y in b {axes.push(x.cross(y));}}
    let mut best=f64::INFINITY;
    for axis in axes {
        let length=axis.length();if length<1e-12{continue;}
        let mut n=axis/length;if n.dot(up)<0.{n=-n;}
        let rate=n.dot(up);if rate<1e-12{continue;}
        let radius=path.half.dot((path.pose.rotation.inverse()*n).abs())+other_half.dot((other.rotation.inverse()*n).abs());
        let distance=(radius-allowed-(path.pose.translation-other.translation).dot(n))/rate;
        best=best.min(distance.max(0.));
    }
    best
}

/// For each rotation interval, every interpolated box is contained in this midpoint box.
/// Dimensions interpolate linearly in feet-local coordinates (Y ranges from0 to2*halfY).
/// Rodrigues' formula bounds rotation deviation on each local axis; an invariant yaw/up
/// axis has exactly zero rotational inflation, allowing a clear grounded yaw transition.
fn envelopes(feet:Vec3,old:&Pose,old_half:Vec3,new:&Pose,new_half:Vec3)->Result<Vec<Envelope>,String>{
    let mut destination=new.rotation;
    if old.rotation.dot(destination)<0.{destination=-destination;}
    let rotation=destination*old.rotation.inverse();let (axis,angle)=rotation.to_axis_angle();
    if !angle.is_finite(){return Err("nonfinite feet rotation".into());}
    let count=if angle.abs()<1e-12 {1}else{MAX_INTERVALS};
    let mut result=Vec::with_capacity(count);
    for i in 0..count {
        let a=i as f64/count as f64;let b=(i+1) as f64/count as f64;
        let rotation=old.rotation.slerp(destination,(a+b)*0.5);
        let h=(old_half+(new_half-old_half)*a).max(old_half+(new_half-old_half)*b);
        let local_axis=rotation.inverse()*axis;
        let delta=angle.abs()/(2.*count as f64);
        let center=Vec3::new(0.,h.y,0.);
        let support=|direction:Vec3|center.dot(direction).abs()+h.dot(direction.abs());
        let mut margin=Vec3::ZERO;
        for j in 0..3 {
            let mut e=Vec3::ZERO;e[j]=1.;
            margin[j]=(1.-delta.cos())*support(e-local_axis*local_axis[j])+delta.sin()*support(e.cross(local_axis));
        }
        // No isotropic epsilon inflation: that would manufacture penetration into a touching
        // floor. The contact comparison has the explicit roundoff allowance below.
        let half=h+margin;let pose=Pose {translation:feet+rotation*center,rotation};
        let aabb=SharedShape::cuboid(half.x,half.y,half.z).compute_aabb(&pose);
        bounded(aabb.mins)?;bounded(aabb.maxs)?;
        result.push(Envelope {pose,half,aabb,translation:None});
    }
    Ok(result)
}

fn contact_depth(pose:&Pose,shape:&dyn rapier3d_f64::parry::shape::Shape,other_pose:&Pose,other:&dyn rapier3d_f64::parry::shape::Shape)->Result<f64,String>{
    Ok(rapier3d_f64::parry::query::contact(pose,shape,other_pose,other,0.)
        .map_err(|_|"unsupported feet transition contact primitive")?.map_or(0.,|c|(-c.dist).max(0.)))
}

pub(super) fn dispatch(registry:&mut Registry,scene:i64,op:i32,ids:&[i64],values:&[f64])->Result<Vec<i64>,String>{
    if op==90 {require(values,0)?;if !ids.is_empty(){return Err("feet pose capability takes no identities".into());}return Ok(vec![1,MAX_INTERVALS as i64,MAX_COLLIDERS as i64,MAX_QUERIES as i64]);}
    if (op!=91&&op!=92)||ids.len()!=10{return Err("feet transition requires lease and exact clock/sequence".into());}require(values,23)?;
    let (key,mass,before_identity)=controlled::pose_actor(registry,scene,&ids[..7])?;
    controlled::transfer_ready(registry,scene,scene)?;
    let region=transfer::lookup(registry,scene)?;
    if ids[7]!=region.time_nanos||ids[8]!=region.mutation||ids[9]!=before_identity[9] {
        return Err("feet transition stale scene clock or consumed input".into());
    }
    if region.sim.collider_set.len()>MAX_COLLIDERS{return Err("feet transition collider scan capacity".into());}
    let handle=region.bodies[&key];let body=&region.sim.rigid_body_set[handle];
    if body.colliders().len()!=1||!body.is_dynamic(){return Err("feet transition requires one dynamic controlled box".into());}
    let collider_handle=body.colliders()[0];let collider=&region.sim.collider_set[collider_handle];
    let local=collider.position_wrt_parent().ok_or("controlled collider parent missing")?;
    if local.translation!=Vec3::ZERO||local.rotation.dot(Pose::IDENTITY.rotation).abs()<1.-1e-12 {
        return Err("feet transition requires centered collider".into());
    }
    let old_pose=*body.position();let old_half=collider.shape().as_cuboid().ok_or("feet transition requires cuboid")?.half_extents;
    let expected_pose=transfer::pose(&values[..7])?;let expected_half=half(values,7)?;
    let velocity=body.linvel();let angular=body.angvel();
    if !same_pose(&old_pose,&expected_pose)||(old_half-expected_half).length()>EPS
        ||velocity!=vec(values,10)?||angular!=vec(values,13)? {
        return Err("feet transition actual pose/shape/velocity CAS mismatch".into());
    }
    if region.sim.impulse_joint_set.iter().any(|(_,joint)|joint.body1==handle||joint.body2==handle)
        ||region.sim.multibody_joint_set.rigid_body_link(handle).is_some(){return Err("feet transition cannot alter constrained body pose".into());}
    let next_half=half(values,16)?;
    let feet=old_pose.translation-old_pose.rotation*Vec3::new(0.,old_half.y,0.);
    let q=transfer::pose(&[0.,0.,0.,values[19],values[20],values[21],values[22]])?.rotation;
    let mut next_pose=Pose {translation:feet+q*Vec3::new(0.,next_half.y,0.),rotation:q};
    let mut paths=envelopes(feet,&old_pose,old_half,&next_pose,next_half)?;
    let mut coverage=paths[0].aabb;for path in &paths[1..]{coverage.mins=coverage.mins.min(path.aabb.mins);coverage.maxs=coverage.maxs.max(path.aabb.maxs);}
    // Discovery includes every permitted projection/lift. This is only broadphase; the exact
    // admitted route below has separate continuous envelopes and original-primitive checks.
    let discovery=if op==92 {Aabb::new(coverage.mins-Vec3::splat(MAX_CORRECTION),coverage.maxs+Vec3::splat(MAX_CORRECTION))}else{coverage};
    let mut queries=0usize;let mut candidates=0usize;
    let old_shape=SharedShape::cuboid(old_half.x,old_half.y,old_half.z);
    let mut primitives=Vec::new();
    for (other_handle,other) in region.sim.collider_set.iter(){
        if other_handle==collider_handle||!other.is_enabled(){continue;}
        let other_pose=if let Some(parent)=other.parent(){*region.sim.rigid_body_set[parent].position()*other.position_wrt_parent().ok_or("foreign collider parent pose missing")?}else{*other.position()};
        if !discovery.intersects(&other.shape().compute_aabb(&other_pose)){continue;}
        if other.is_sensor(){continue;}
        candidates+=1;
        let mut collect=|primitive_pose:Pose,primitive:&SharedShape|->Result<(),String>{
            if primitive.as_cuboid().is_none(){return Err("unsupported feet transition collider primitive".into());}
            if primitives.len()>=MAX_QUERIES/(MAX_INTERVALS+12){return Err("feet transition primitive collection capacity".into());}
            charge(&mut queries)?;
            let previous=contact_depth(&old_pose,old_shape.as_ref(),&primitive_pose,primitive.as_ref())?;
            if previous>0.005+EPS{return Err("feet transition starts in excessive solver penetration".into());}
            primitives.push((primitive_pose,primitive.clone(),previous));
            Ok(())
        };
        if let Some(compound)=other.shape().as_compound(){
            if compound.shapes().len()>4096{return Err("feet transition compound capacity".into());}
            for (local,primitive) in compound.shapes(){collect(other_pose*local,primitive)?;}
        }else{collect(other_pose,other.shared_shape())?;}
    }
    let mut correction=Vec3::ZERO;let mut correction_distance=0.;let mut lift=0.;
    if op==92 {
        let shape=SharedShape::cuboid(next_half.x,next_half.y,next_half.z);
        // Match the existing Java controller's bounded deepest-normal endpoint proposal.
        // This computes a destination only; it does NOT authorize the path to that pose.
        for iteration in 0..8 {
            let mut deepest:Option<(f64,Vec3)>=None;
            for (pose,primitive,_) in &primitives {
                charge(&mut queries)?;
                if let Some(contact)=rapier3d_f64::parry::query::contact(&next_pose,shape.as_ref(),pose,primitive.as_ref(),0.)
                    .map_err(|_|"unsupported projection contact primitive")? {
                    let depth=(-contact.dist).max(0.);
                    if depth>EPS&&deepest.as_ref().map_or(true,|(d,_)|depth>*d){deepest=Some((depth,-contact.normal1));}
                }
            }
            let Some((depth,normal))=deepest else{break};
            let amount=depth+PROJECTION_SKIN;correction_distance+=amount;
            if correction_distance>MAX_CORRECTION||iteration==7{return Err("feet transition bounded projection exhausted".into());}
            let shift=normal*amount;correction+=shift;next_pose.translation+=shift;
        }
        let up=old_pose.rotation*Vec3::Y;
        // Find a clearance lift for the complete rotational envelope relative to original
        // feet. The subsequent real-collider checks prove this proposed detour is clear.
        for path in &paths {
            let radius=path.half.dot((path.pose.rotation.inverse()*up).abs());
            lift=f64::max(lift,radius-(path.pose.translation-feet).dot(up));
            for (pose,primitive,previous) in &primitives {
                charge(&mut queries)?;
                lift=lift.max(clearance_lift(path,pose,primitive.as_cuboid().unwrap().half_extents,up,*previous));
            }
        }
        lift=f64::max(lift,0.);
        if lift>0.{lift+=PROJECTION_SKIN;}
        if lift>MAX_CORRECTION{return Err("feet transition clearance lift exceeds bound".into());}
        let offset=up*lift;
        let lifted_old=Pose {translation:old_pose.translation+offset,..old_pose};
        let lifted_next=Pose {translation:feet+offset+q*Vec3::new(0.,next_half.y,0.),rotation:q};
        let mut route=Vec::with_capacity(paths.len()+2);
        route.push(translation_envelope(old_pose,lifted_old,old_half));
        for path in &mut paths {path.pose.translation+=offset;path.aabb.mins+=offset;path.aabb.maxs+=offset;}
        route.extend(paths);
        route.push(translation_envelope(lifted_next,next_pose,next_half));
        paths=route;
    }
    coverage=paths[0].aabb;for path in &paths[1..]{coverage.mins=coverage.mins.min(path.aabb.mins);coverage.maxs=coverage.maxs.max(path.aabb.maxs);}
    bounded(coverage.mins)?;bounded(coverage.maxs)?;
    for (primitive_pose,primitive,previous) in &primitives {
        // Preserve only this exact primitive's existing solver penetration. A proposed
        // projection is never permission for new/deeper penetration on ANY route segment.
        for path in &paths {
            charge(&mut queries)?;
            if let Some((start,end,half))=&path.translation {
                if translation_clear(start,end,*half,primitive_pose,primitive.as_cuboid().unwrap().half_extents,*previous){continue;}
            }
            let shape=SharedShape::cuboid(path.half.x,path.half.y,path.half.z);
            if contact_depth(&path.pose,shape.as_ref(),primitive_pose,primitive.as_ref())?>*previous+EPS {
                return Err(format!("feet transition swept envelope intersects collision: translation={} depth={} previous={} lift={} half={:?} pose={:?}",path.translation.is_some(),contact_depth(&path.pose,shape.as_ref(),primitive_pose,primitive.as_ref())?,previous,lift,path.half,path.pose));
            }
        }
    }
    let mutation=region.mutation.checked_add(1).ok_or("feet transition mutation exhausted")?;
    let radius=next_half.length()+region.sim.parameters.prediction_distance()+0.02*region.sim.parameters.length_unit;
    let shape=SharedShape::cuboid(next_half.x,next_half.y,next_half.z);
    let region=registry.scenes.get_mut(&scene).unwrap();
    // This is also the native stage-only gate. Authoritative bodies have no active collector.
    region.sim.rigid_body_set[handle].planetary_expand_sweep(radius,&coverage).map_err(|e|e.to_string())?;
    region.sim.collider_set[collider_handle].set_shape(shape);
    region.sim.collider_set[collider_handle].set_mass(mass);
    region.sim.rigid_body_set[handle].set_position(next_pose,true);
    region.sim.rigid_body_set[handle].recompute_mass_properties_from_colliders(&region.sim.collider_set);
    region.mutation=mutation;
    let after_identity=controlled::pose_publish_identity(registry,scene,ids[0])?;
    let mut out=vec![1];out.extend(before_identity);out.extend(after_identity);
    out.extend([scene,ids[4],ids[7],ids[8],scene,ids[4],ids[7],mutation]);
    append_vec(&mut out,feet);append_pose(&mut out,&old_pose);append_vec(&mut out,old_half);
    append_pose(&mut out,&next_pose);append_vec(&mut out,next_half);append_vec(&mut out,velocity);append_vec(&mut out,angular);
    append_vec(&mut out,coverage.mins);append_vec(&mut out,coverage.maxs);out.extend([paths.len() as i64,candidates as i64,queries as i64]);
    if op==92 {append_vec(&mut out,correction);out.extend([correction_distance.to_bits() as i64,lift.to_bits() as i64]);}Ok(out)
}


