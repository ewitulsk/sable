//! Unrelated scenes keep advancing while a preview owns its source. Their speculative
//! contact workspace is bounded too; a failed guard never modifies authoritative actors.
use super::*;
use rapier3d_f64::geometry::PlanetaryContactGuard;

pub(super) fn advance(registry:&mut Registry,scene:i64,nanos:i64)->Result<(),String> {
    if nanos<=0||nanos>50_000_000 {return Err("invalid native step duration".into());}
    if registry.preview.is_none()||preview::retains(registry,scene) {
        return Err("unrelated guarded step requires another retained preview".into());
    }
    if transfer::mapping_retains(registry,scene) {
        return Err("mapped transfer receipt retains native scene step".into());
    }
    let source=transfer::lookup(registry,scene)?;
    source.time_nanos.checked_add(nanos).ok_or("simulation clock exhausted")?;
    source.mutation.checked_add(1).ok_or("scene mutation exhausted")?;
    // The live source remains allocated throughout integration, as does the other preview.
    // Charge this additional complete clone before constructing any physical arenas.
    transfer::preview_budget(registry,&source.sim)?;
    let mut sections=HashMap::with_capacity(source.sections.len());
    for (id,s) in &source.sections {
        sections.insert(*id,Section {revision:s.revision,body:s.body,collider:s.collider,
            _parts:PartBudget::reserve(s._parts.0)?,fingerprint:s.fingerprint,
            translation:s.translation,resident:s.resident});
    }
    let candidate=Region {
        terrain_batch:source.terrain_batch.clone(),sim:source.sim.staged_clone(),sections,
        bodies:source.bodies.clone(),epoch:source.epoch,failed_range:source.failed_range,
        streamed_terrain:source.streamed_terrain,section_high_water:source.section_high_water,
        mutation:source.mutation,time_nanos:source.time_nanos,body_epochs:source.body_epochs.clone(),
        body_history:source.body_history.clone(),joints:source.joints.clone(),
        next_legacy_joint:source.next_legacy_joint,
    };
    let mut staged=Registry::default();
    staged.controlled=controlled::snapshot_scene(&registry.controlled,scene);
    staged.scenes.insert(scene,candidate);
    // No exclusion: all authoritative scenes and the retained preview still consume memory.
    let guard=PlanetaryContactGuard::new(transfer::remaining_contact_budget(registry,None)?);
    let sim=&mut staged.scenes.get_mut(&scene).unwrap().sim;
    sim.flush_pending_removals();
    sim.narrow_phase.planetary_clear_contact_guard();
    sim.narrow_phase.planetary_guard_contacts(&sim.collider_set,&guard)?;
    advance_region_core(&mut staged,scene,nanos)?;
    guard.snapshot()?;
    transfer::preview_budget(registry,&staged.scenes[&scene].sim)?;
    // Every fallible operation finished against detached state. Publish only this scene and
    // its exact controlled actor slice, retaining all other actors and the selected preview.
    let mut ready=staged.scenes.remove(&scene).unwrap();
    ready.sim.narrow_phase.planetary_clear_contact_guard();
    ready.sim.pipeline=PhysicsPipeline::new();
    registry.scenes.insert(scene,ready);
    controlled::publish_scene(&mut registry.controlled,scene,staged.controlled);
    Ok(())
}
