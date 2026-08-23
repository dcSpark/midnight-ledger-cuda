// Copyright 2026 Midnight Foundation
// SPDX-License-Identifier: Apache-2.0

import Foundation
import Metal

struct BenchParams {
    var threadCount: UInt32
    var iterations: UInt32
}

struct SplitMix64 {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        return z ^ (z >> 31)
    }
}

@inline(__always)
func rotateLeft(_ value: UInt32, _ amount: UInt32) -> UInt32 {
    (value << amount) | (value >> (32 - amount))
}

@inline(__always)
func cpuStep(_ state: SIMD2<UInt32>) -> SIMD2<UInt32> {
    let product = UInt64(state.x) * UInt64(state.y)
    let low = UInt32(truncatingIfNeeded: product)
    let high = UInt32(truncatingIfNeeded: product >> 32)
    let next = (low ^ rotateLeft(high, 13)) &+ 0x243f6a88
    return SIMD2(state.y, next)
}

func makeBuffer<T>(_ device: MTLDevice, values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { bytes in
        device.makeBuffer(
            bytes: bytes.baseAddress!,
            length: values.count * MemoryLayout<T>.stride,
            options: .storageModeShared)!
    }
}

func bufferArray<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func evenPopulationMedian(_ sorted: [Double]) -> Double {
    precondition(!sorted.isEmpty)
    let upper = sorted.count / 2
    if sorted.count % 2 == 1 {
        return sorted[upper]
    }
    return (sorted[upper - 1] + sorted[upper]) / 2.0
}

func dispatch(
    queue: MTLCommandQueue,
    pipeline: MTLComputePipelineState,
    input: MTLBuffer,
    output: MTLBuffer,
    params: MTLBuffer,
    threads: Int
) -> (gpuNS: UInt64, wallNS: UInt64) {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBuffer(params, offset: 0, index: 2)
    encoder.dispatchThreads(
        MTLSize(width: threads, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
            width: min(256, pipeline.maxTotalThreadsPerThreadgroup),
            height: 1,
            depth: 1))
    encoder.endEncoding()
    let wallStart = DispatchTime.now().uptimeNanoseconds
    command.commit()
    command.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds
    guard command.status == .completed else {
        fatalError("Metal command failed: \(command.error?.localizedDescription ?? "unknown")")
    }
    return (
        UInt64((command.gpuEndTime - command.gpuStartTime) * 1_000_000_000.0),
        wallEnd - wallStart)
}

guard CommandLine.arguments.count == 2 else {
    fatalError("usage: rotating-mulhi-probe /path/to/rotating.metallib")
}
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal unavailable")
}
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let queue = device.makeCommandQueue()!
let mulhi = try device.makeComputePipelineState(
    function: library.makeFunction(name: "bench_rotating_mulhi")!)
let widened = try device.makeComputePipelineState(
    function: library.makeFunction(name: "bench_rotating_widened")!)

print("device=\(device.name)")
print("registry_id=\(device.registryID)")
print("thread_execution_width_mulhi=\(mulhi.threadExecutionWidth)")
print("thread_execution_width_widened=\(widened.threadExecutionWidth)")

var rng = SplitMix64(state: 0x510e527fade682d1)
let oracleThreads = 257
let oracleIterations: UInt32 = 37
var oracleSeed: [SIMD2<UInt32>] = []
for _ in 0..<oracleThreads {
    oracleSeed.append(SIMD2(
        UInt32(truncatingIfNeeded: rng.next()),
        UInt32(truncatingIfNeeded: rng.next())))
}
let oracleInput = makeBuffer(device, values: oracleSeed)
let oracleMulhi = device.makeBuffer(
    length: oracleThreads * MemoryLayout<SIMD2<UInt32>>.stride,
    options: .storageModeShared)!
let oracleWidened = device.makeBuffer(
    length: oracleThreads * MemoryLayout<SIMD2<UInt32>>.stride,
    options: .storageModeShared)!
var oracleParamsValue = BenchParams(
    threadCount: UInt32(oracleThreads),
    iterations: oracleIterations)
let oracleParams = device.makeBuffer(
    bytes: &oracleParamsValue,
    length: MemoryLayout<BenchParams>.stride,
    options: .storageModeShared)!
_ = dispatch(
    queue: queue, pipeline: mulhi, input: oracleInput, output: oracleMulhi,
    params: oracleParams, threads: oracleThreads)
_ = dispatch(
    queue: queue, pipeline: widened, input: oracleInput, output: oracleWidened,
    params: oracleParams, threads: oracleThreads)
let oracleMulhiValues = bufferArray(
    oracleMulhi, count: oracleThreads, as: SIMD2<UInt32>.self)
let oracleWidenedValues = bufferArray(
    oracleWidened, count: oracleThreads, as: SIMD2<UInt32>.self)
for index in 0..<oracleThreads {
    var expected = oracleSeed[index]
    for _ in 0..<oracleIterations {
        expected = cpuStep(expected)
    }
    guard oracleMulhiValues[index] == expected,
          oracleWidenedValues[index] == expected else {
        fatalError("oracle mismatch at \(index)")
    }
}
print("oracle_threads=\(oracleThreads)")
print("oracle_iterations=\(oracleIterations)")

let benchmarkThreads = 131_072
let benchmarkIterations: UInt32 = 262_144
let samples = 20
var seeds: [SIMD2<UInt32>] = []
seeds.reserveCapacity(benchmarkThreads)
for _ in 0..<benchmarkThreads {
    seeds.append(SIMD2(
        UInt32(truncatingIfNeeded: rng.next()),
        UInt32(truncatingIfNeeded: rng.next())))
}
let input = makeBuffer(device, values: seeds)
let outputMulhi = device.makeBuffer(
    length: benchmarkThreads * MemoryLayout<SIMD2<UInt32>>.stride,
    options: .storageModeShared)!
let outputWidened = device.makeBuffer(
    length: benchmarkThreads * MemoryLayout<SIMD2<UInt32>>.stride,
    options: .storageModeShared)!
var paramsValue = BenchParams(
    threadCount: UInt32(benchmarkThreads),
    iterations: benchmarkIterations)
let params = device.makeBuffer(
    bytes: &paramsValue,
    length: MemoryLayout<BenchParams>.stride,
    options: .storageModeShared)!

for _ in 0..<3 {
    _ = dispatch(
        queue: queue, pipeline: mulhi, input: input, output: outputMulhi,
        params: params, threads: benchmarkThreads)
    _ = dispatch(
        queue: queue, pipeline: widened, input: input, output: outputWidened,
        params: params, threads: benchmarkThreads)
}

var mulhiGPU: [UInt64] = []
var widenedGPU: [UInt64] = []
var mulhiWall: [UInt64] = []
var widenedWall: [UInt64] = []
for sample in 0..<samples {
    let mulhiFirst = sample % 2 == 0
    let first = dispatch(
        queue: queue,
        pipeline: mulhiFirst ? mulhi : widened,
        input: input,
        output: mulhiFirst ? outputMulhi : outputWidened,
        params: params,
        threads: benchmarkThreads)
    let second = dispatch(
        queue: queue,
        pipeline: mulhiFirst ? widened : mulhi,
        input: input,
        output: mulhiFirst ? outputWidened : outputMulhi,
        params: params,
        threads: benchmarkThreads)
    let m = mulhiFirst ? first : second
    let w = mulhiFirst ? second : first
    mulhiGPU.append(m.gpuNS)
    widenedGPU.append(w.gpuNS)
    mulhiWall.append(m.wallNS)
    widenedWall.append(w.wallNS)
    print("sample=\(sample) order=\(mulhiFirst ? "mulhi,widened" : "widened,mulhi") mulhi_gpu_ns=\(m.gpuNS) widened_gpu_ns=\(w.gpuNS) mulhi_wall_ns=\(m.wallNS) widened_wall_ns=\(w.wallNS)")
}

let finalMulhi = bufferArray(
    outputMulhi, count: benchmarkThreads, as: SIMD2<UInt32>.self)
let finalWidened = bufferArray(
    outputWidened, count: benchmarkThreads, as: SIMD2<UInt32>.self)
guard finalMulhi == finalWidened else {
    fatalError("observable benchmark outputs differ")
}
var checksum: UInt64 = 0
for value in finalMulhi {
    checksum &+= UInt64(value.x) &+ UInt64(value.y)
}

let products = Double(benchmarkThreads) * Double(benchmarkIterations)
let mulhiRates = mulhiGPU.map { products * 1_000_000_000.0 / Double($0) }.sorted()
let widenedRates = widenedGPU.map { products * 1_000_000_000.0 / Double($0) }.sorted()
let ratios = zip(mulhiGPU, widenedGPU).map { Double($0.1) / Double($0.0) }.sorted()
print("benchmark_output_checksum=\(checksum)")
print("benchmark_threads=\(benchmarkThreads)")
print("benchmark_iterations=\(benchmarkIterations)")
print("complete_products_per_sample=\(UInt64(products))")
print("mulhi_median_gpu_ns=\(evenPopulationMedian(mulhiGPU.map { Double($0) }.sorted()))")
print("mulhi_median_complete_products_per_s=\(evenPopulationMedian(mulhiRates))")
print("mulhi_empirical_p5_complete_products_per_s=\(mulhiRates[0])")
print("widened_median_gpu_ns=\(evenPopulationMedian(widenedGPU.map { Double($0) }.sorted()))")
print("widened_median_complete_products_per_s=\(evenPopulationMedian(widenedRates))")
print("widened_empirical_p5_complete_products_per_s=\(widenedRates[0])")
print("paired_widened_time_over_mulhi_time_median=\(evenPopulationMedian(ratios))")
print("paired_widened_time_over_mulhi_time_min=\(ratios[0])")
print("paired_widened_time_over_mulhi_time_max=\(ratios[samples - 1])")
