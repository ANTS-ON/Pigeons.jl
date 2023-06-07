struct HopsyTarget <: StreamTarget
    command::Cmd
end

initialization(target::HopsyTarget, rng::SplittableRandom, replica_index::Int64) = 
    StreamState(
        `$(target.command) --seed $(42)`,
        replica_index)

hopsy_toy_mvn(file_path::String) =
    HopsyTarget(`python $file_path`)