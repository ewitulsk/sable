//! Explicit live native history. Clock labels are not integrated duration.
//! Dormant-save provenance belongs to the durable owner; these counters begin at actual allocation.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BodyHistory {
    pub domain: i64,
    pub start: i64,
    pub end: i64,
    pub integrated: i64,
}
impl BodyHistory {
    pub fn allocated(domain: i64, now: i64) -> Self {
        Self {
            domain,
            start: now,
            end: now,
            integrated: 0,
        }
    }
    fn advanced(self, domain: i64, start: i64, end: i64) -> Result<Self, String> {
        let duration = end
            .checked_sub(start)
            .ok_or("native history interval overflow")?;
        if domain <= 0 || start < 0 || duration <= 0 {
            return Err("invalid native history interval".into());
        }
        Ok(Self {
            domain,
            start,
            end,
            integrated: self
                .integrated
                .checked_add(duration)
                .ok_or("native integrated history exhausted")?,
        })
    }
    pub fn words(self) -> [i64; 4] {
        [self.domain, self.start, self.end, self.integrated]
    }
}

/// Precompute before advancing authoritative physics: allocation/arithmetic failure cannot leave
/// the physical step ahead of its retained history. This is bounded by the actual body registry.
pub(super) fn prepare_advance(
    region: &Region,
    domain: i64,
    end: i64,
) -> Result<HashMap<i64, BodyHistory>, String> {
    if region.body_history.len() != region.bodies.len()
        || region
            .body_history
            .keys()
            .any(|id| !region.bodies.contains_key(id))
    {
        return Err("native body/history ownership mismatch".into());
    }
    region
        .body_history
        .iter()
        .map(|(id, history)| Ok((*id, history.advanced(domain, region.time_nanos, end)?)))
        .collect()
}

pub(super) fn map_eligibility(
    source_now: i64,
    destination_now: i64,
    source_next: i64,
) -> Result<i64, String> {
    if source_now < 0 || destination_now < 0 || source_next < 0 {
        return Err("negative clock mapping input".into());
    }
    let remaining = source_next
        .checked_sub(source_now)
        .ok_or("eligibility duration overflow")?
        .max(0);
    destination_now
        .checked_add(remaining)
        .ok_or_else(|| "mapped eligibility overflow".into())
}
