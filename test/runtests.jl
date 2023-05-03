using Pigeons

using ArgMacros
using Distributions
using DynamicPPL
using LinearAlgebra
using MPI
using MPIPreferences
using OnlineStats
using Random
using Serialization
using SplittableRandoms
using Statistics
using Test

import Pigeons: my_global_indices, LoadBalance, my_load,
                find_process, split_slice

include("misc.jl")
include("slice_sampler_test.jl")
include("var_reference_test.jl")
include("turing.jl")
include("vector.jl")

function test_load_balance(n_processes, n_tasks)
    for p in 1:n_processes
        lb = LoadBalance(p, n_processes, n_tasks)        
        globals = my_global_indices(lb)
        @assert length(globals) == my_load(lb)
        for g in globals
            @assert find_process(lb, g) == p
        end
    end
end

# @testset "Allocs-HMC" begin
#     allocs_10_rounds = Pigeons.last_round_max_allocation(pigeons(n_rounds = 10, target = toy_mvn_target(1), explorer = HMC()))
#     allocs_11_rounds = Pigeons.last_round_max_allocation(pigeons(n_rounds = 11, target = toy_mvn_target(1), explorer = HMC()))
#     @test allocs_10_rounds == allocs_11_rounds
# end

hmc(target, std_devs = nothing) =
    pigeons(; 
        target, 
        explorer = Pigeons.staticHMC(0.2, 1.0, 3, std_devs), 
        n_chains = 2, n_rounds = 10, recorder_builders = Pigeons.online_recorder_builders())

mean_mh_accept(pt) = mean(Pigeons.explorer_mh_prs(pt))

@testset "HMC dimensional autoscale" begin
    for i in 0:3
        d = 10^i
        @test mean_mh_accept(hmc(toy_mvn_target(d))) > 0.98
    end
end

@testset "Check HMC pre-conditioning" begin
    tol = 1e-5
    iso = Pigeons.HetPrecisionNormalLogPotential(2) 
    before = mean_mh_accept(hmc(iso))

    bad_conditioning_target = Pigeons.HetPrecisionNormalLogPotential([50.0, 1.0])
    bad = mean_mh_accept(hmc(bad_conditioning_target))
    @test abs(before - bad) > tol

    std_devs = 1.0 ./ sqrt.(bad_conditioning_target.precisions)
    corrected = mean_mh_accept(hmc(bad_conditioning_target, std_devs))
    @test before == corrected
end

@testset "Check HMC involution" begin
    rng = SplittableRandom(1)

    my_target = Pigeons.HetPrecisionNormalLogPotential([5.0, 1.1]) 
    some_cond = [2.3, 0.8]

    x = randn(rng, 2)
    v = randn(rng, 2)

    n_leaps = 40

    start = copy(x)
    @test Pigeons.hamiltonian_dynamics!(my_target, some_cond, x, v, 0.1, n_leaps, nothing)
    @test !(x ≈ start)
    @test Pigeons.hamiltonian_dynamics!(my_target, some_cond, x, -v, 0.1, n_leaps, nothing)
    @test x ≈ start
end

@testset "Curvature estimation check" begin
    rng = SplittableRandom(1)

    estimated = 5.0
    residual = 1.1
    my_target = Pigeons.HetPrecisionNormalLogPotential([estimated, residual]) 
    # say we are able to capture part of the shape of the 
    # target (here, first component), can we estimate residual?
    partly_estimated_std_devs = [1.0 / sqrt(estimated), 1.0]

    x = randn(rng, 2)
    n_leaps = 40
    recorders = (; directional_second_derivatives =  GroupBy(Int, Extrema()))
    replica = Pigeons.Replica(nothing, 1, rng, recorders, 1)

    v = randn(rng, 2)
    for i in 1:100
        Pigeons.hamiltonian_dynamics!(my_target, partly_estimated_std_devs, x, v, 0.1, n_leaps, replica)
        v = randn(rng, 2)
    end

    @test maximum(replica.recorders.directional_second_derivatives[1]) ≈ residual
end

@testset "Allocs" begin
    allocs_10_rounds = Pigeons.last_round_max_allocation(pigeons(n_rounds = 10, target = toy_mvn_target(100)))
    allocs_11_rounds = Pigeons.last_round_max_allocation(pigeons(n_rounds = 11, target = toy_mvn_target(100)))
    @test allocs_10_rounds == allocs_11_rounds
end

@testset "Variational reference" begin
    test_var_reference()
end

@testset "Traces" begin
    pt = pigeons(target = toy_mvn_target(10), recorder_builders = [traces, disk], checkpoint = true) 
    @test length(pt.reduced_recorders.traces) == 1024 
    marginal = [get_sample(pt, 10, i)[1] for i in 1:1024]
    @test abs(mean(marginal) - 0.0) < 0.05 

    # check that the disk serialization gives the same result
    process_samples(pt) do chain, scan, sample 
        @test sample == get_sample(pt, chain, scan)
    end
end

@testset "Examples directory" begin
    # make sure the examples run correctly
    include("../examples/custom-path.jl")
    include("../examples/general-target.jl")
end

@testset "Check sources can be sorted automatically" begin
    cd("..") do
        Pigeons.sort_includes("Pigeons.jl")
    end
end

@testset "MPI backend" begin
    @info "MPI: using $(MPIPreferences.abi) ($(MPIPreferences.binary))"
    if haskey(ENV,"JULIA_MPI_TEST_BINARY")
        @test ENV["JULIA_MPI_TEST_BINARY"] == MPIPreferences.binary
    end
    if haskey(ENV,"JULIA_MPI_TEST_ABI")
        @test ENV["JULIA_MPI_TEST_ABI"] == MPIPreferences.abi
    end
end

@testset "GC+multithreading" begin
    mpi_test(2, "gc_test.jl")
end

@testset "Stepping stone" begin
    pt = pigeons(target = toy_mvn_target(100));
    p = stepping_stone_pair(pt)
    truth = Pigeons.analytic_lognormalization(toy_mvn_target(100))
    @test abs(p[1] - truth) < 1
    @test abs(p[2] - truth) < 1
end

@testset "Round trips" begin
    n_chains = 4
    n_rounds = 5
    
    pt = pigeons(; target = Pigeons.TestSwapper(1.0), recorder_builders = [Pigeons.round_trip], n_chains, n_rounds);
    
    len = 2^(n_rounds)
    truth = 0.0
    for i in 0:(n_chains-1)
        truth += floor(max(len - i, 0) / n_chains / 2)
    end

    @test truth == Pigeons.n_round_trips(pt)
end

@testset "Moments" begin
    pt = pigeons(target = toy_mvn_target(2), recorder_builders = [Pigeons.target_online], n_rounds = 20);
    for var_name in Pigeons.continuous_variables(pt)
        m = mean(pt, var_name)
        for i in eachindex(m)
            @test abs(m[i] - 0.0) < 0.001
        end
        v = var(pt, var_name) 
        for i in eachindex(v) 
            @test abs(v[i] - 0.1) < 0.001 
        end
    end
end



@testset "Parallelism Invariance" begin
    n_mpis = set_n_mpis_to_one_on_windows(4)
    recorder_builders = [swap_acceptance_pr, index_process, log_sum_ratio, round_trip, energy_ac1]

    # test swapper 
    pigeons(
        target = toy_mvn_target(1), 
        n_rounds = 10,
        checked_round = 3, 
        recorder_builders = recorder_builders,
        checkpoint = true, 
        on = ChildProcess(
                n_local_mpi_processes = n_mpis,
                n_threads = 2,
                mpiexec_args = extra_mpi_args())) 

    # Turing:
    pigeons(
        target = TuringLogPotential(flip_model_unidentifiable()), 
        n_rounds = 10,
        checked_round = 3, 
        multithreaded = true,
        recorder_builders = recorder_builders,
        checkpoint = true, 
        on = ChildProcess(
                dependencies = [Distributions, DynamicPPL, LinearAlgebra, "turing.jl"],
                n_local_mpi_processes = n_mpis,
                n_threads = 2,
                mpiexec_args = extra_mpi_args()))

    # Blang:
    if !Sys.iswindows() # JNI crashes on windows; see commit right after c016f59c84645346692f720854b7531743c728bf
        Pigeons.setup_blang("blangDemos")
        pigeons(; 
            target = Pigeons.blang_ising(), 
            n_rounds = 10,
            checked_round = 3, 
            recorder_builders = recorder_builders, 
            multithreaded = true, 
            checkpoint = true, 
            on = ChildProcess(
                    n_local_mpi_processes = n_mpis,
                    n_threads = 2,
                    mpiexec_args = extra_mpi_args()))
    end
end

@testset "Longer MPI" begin
    n_mpis = set_n_mpis_to_one_on_windows(4)
    recorder_builders = []
    pigeons(
        target = Pigeons.TestSwapper(0.5), 
        n_rounds = 14,
        checked_round = 12, 
        n_chains = 200,
        multithreaded = false,
        recorder_builders = recorder_builders,
        checkpoint = true, 
        on = ChildProcess(
                n_local_mpi_processes = n_mpis,
                n_threads = 2,
                mpiexec_args = extra_mpi_args())) 
end

@testset "Entanglement" begin
    mpi_test(1, "entanglement_test.jl")
    mpi_test(2, "entanglement_test.jl")

    mpi_test(1, "reduce_test.jl")
    mpi_test(2, "reduce_test.jl")
    mpi_test(3, "reduce_test.jl")
end

@testset "PermutedDistributedArray" begin
    mpi_test(1, "permuted_test.jl", options = ["-s"])
    mpi_test(1, "permuted_test.jl")
    mpi_test(2, "permuted_test.jl")
end

@testset "LoadBalance" begin
    for i in 1:20
        for j in i:30
            test_load_balance(i, j)
        end
    end
end

@testset "LogSum" begin
    m = Pigeons.LogSum()
    
    fit!(m, 2.1)
    fit!(m, 4)
    v1 = value(m)
    @assert v1 ≈ log(exp(2.1) + exp(4))


    fit!(m, 2.1)
    fit!(m, 4)
    m2 = Pigeons.LogSum() 
    fit!(m2, 50.1)
    combined = merge(m, m2)
    @assert value(combined) ≈ log(exp(v1) + exp(50.1))

    fit!(m, 2.1)
    fit!(m, 4)
    empty!(m)
    @assert value(m) == -Pigeons.inf(0.0)
end

function test_split_slice()
    # test disjoint random streams
    set = Set{Float64}()
    push!(set, test_split_slice_helper(1:10)...)
    push!(set, test_split_slice_helper(11:20)...)
    @test length(set) == 20

    # test overlapping
    set = Set{Float64}()
    push!(set, test_split_slice_helper(1:15)...)
    push!(set, test_split_slice_helper(10:20)...)
    @test length(set) == 20
    return true
end

test_split_slice_helper(range) = [rand(r) for r in split_slice(range,  SplittableRandom(1))]

@testset "split_test" begin
    test_split_slice()
end

@testset "Serialize" begin
    mpi_test(1, "serialization_test.jl")
end

@testset "SliceSampler" begin
    test_slice_sampler()
end


