@testset "Turing-gradient" begin
    target = Pigeons.toy_turing_target()

    logz_mala = stepping_stone_pair(pigeons(; target, explorer = SliceSampler()))
    logz_slicer = stepping_stone_pair(pigeons(; target, explorer = SliceSampler()))

    @test abs(logz_mala[1] - logz_slicer[1]) < 0.1
end