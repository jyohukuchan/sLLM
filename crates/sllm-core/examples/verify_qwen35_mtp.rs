use sllm_core::{
    QwenComponentSelection, WeightClassification, build_verified_qwen_component_weight_load_plan,
    build_verified_qwen35_mtp_manifest, build_verified_qwen35_vision_manifest, read_model_lock,
    verify_model_cache,
};
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let lock_path = PathBuf::from(args.next().ok_or("usage: verify_qwen35_mtp LOCK CACHE")?);
    let cache_path = PathBuf::from(args.next().ok_or("usage: verify_qwen35_mtp LOCK CACHE")?);
    if args.next().is_some() {
        return Err("usage: verify_qwen35_mtp LOCK CACHE".into());
    }
    let lock = read_model_lock(lock_path)?;
    let cache = verify_model_cache(&lock, cache_path)?;
    let manifest = build_verified_qwen35_mtp_manifest(&lock, &cache)?;
    let vision = build_verified_qwen35_vision_manifest(&lock, &cache)?;
    let plan = build_verified_qwen_component_weight_load_plan(
        &lock,
        &cache,
        QwenComponentSelection::TEXT_AND_MTP,
    )?;
    let required_mtp = plan
        .entries
        .iter()
        .filter(|entry| {
            entry.tensor_name.starts_with("mtp.")
                && entry.classification == WeightClassification::Required
                && entry.destination_start.is_some()
        })
        .count();
    if required_mtp != manifest.tensors.len() {
        return Err("component load plan did not require the exact MTP manifest".into());
    }
    let all_plan =
        build_verified_qwen_component_weight_load_plan(&lock, &cache, QwenComponentSelection::ALL)?;
    println!(
        "Qwen3.5 MTP manifest: PASS repo={} revision={} tensors={} resident_bytes={} digest={} plan_digest={} shared={}",
        manifest.repo_id,
        manifest.resolved_revision,
        manifest.tensors.len(),
        manifest.resident_bytes,
        manifest.digest_hex(),
        plan.digest_hex(),
        manifest.shared_embedding,
    );
    println!(
        "Qwen3.5 vision manifest: PASS tensors={} resident_bytes={} digest={} all_plan_digest={} tokens={}/{}/{}/{}",
        vision.tensors.len(),
        vision.resident_bytes,
        vision.digest_hex(),
        all_plan.digest_hex(),
        vision.vision_start_token,
        vision.vision_end_token,
        vision.vision_pad_token,
        vision.image_pad_token,
    );
    Ok(())
}
