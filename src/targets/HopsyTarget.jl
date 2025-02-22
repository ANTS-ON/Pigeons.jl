struct HopsyTarget <: StreamTarget
    command::Cmd
end

initialization(target::HopsyTarget, rng::SplittableRandom, replica_index::Int64) = 
    StreamState(
        `$(target.command) --seed $(split(rng).seed)`,
        replica_index)

hopsy_file_target(file_path::String) =
    HopsyTarget(`python $file_path`)