pub mod aggregate;
pub mod compress;
pub mod convert;
pub mod merge_timestamps;
pub mod purge_cache;
pub mod purge_stale_variants;

use clap::{Args, Subcommand};

#[derive(Args)]
pub struct DataArgs {
    #[command(subcommand)]
    pub command: DataCommand,
}

#[derive(Subcommand)]
pub enum DataCommand {
    /// Consolidate per-trial `result.toml` files into a project workbook.
    Aggregate(aggregate::AggregateArgs),

    /// Archive the run data directory as data.tar.zst.
    Compress(compress::CompressDataArgs),

    /// Batch-convert supported pose files to normalized parquet.
    Convert(convert::ConvertArgs),

    /// Merge per-frame timestamps into authoritative `*-timestamped.parquet` pose files.
    MergeTimestamps(merge_timestamps::MergeTimestampsArgs),

    /// Remove cached stage outputs from the run tree.
    PurgeCache(purge_cache::PurgeCacheArgs),

    /// Remove pre-variant trial outputs after a variant rollout.
    ///
    /// Deletes `output/data/<base>/result.toml` directories whose task type
    /// now has variants configured but whose name lacks the `__<variant>`
    /// suffix. Run after enabling `[[scan.variants]]` /
    /// `[[ingress_profile.tasks.variants]]` so the consolidated workbook
    /// does not mix the legacy single-output sheet with the new per-variant
    /// sheets.
    PurgeStaleVariants(purge_stale_variants::PurgeStaleVariantsArgs),
}

pub fn run(args: DataArgs) -> anyhow::Result<()> {
    match args.command {
        DataCommand::Aggregate(a) => aggregate::run(a),
        DataCommand::Compress(a) => compress::run(a),
        DataCommand::Convert(a) => convert::run(a),
        DataCommand::MergeTimestamps(a) => merge_timestamps::run(a),
        DataCommand::PurgeCache(a) => purge_cache::run(a),
        DataCommand::PurgeStaleVariants(a) => purge_stale_variants::run(a),
    }
}
