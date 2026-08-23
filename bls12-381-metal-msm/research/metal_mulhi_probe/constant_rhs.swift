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
func cpuStep(_ x: UInt32) -> UInt32 {
    let product = UInt64(x) * UInt64(0x9e3779b1 as UInt32)
    let low = UInt32(truncatingIfNeeded: product)
    let high = UInt32(truncatingIfNeeded: product >> 32)
    return (low ^ rotateLeft(high, 13)) &+ 0x243f6a88
}

func bufferArray<T>(_ buffer: MTLBuffer, count: Int, as: T.Type) -> [T] {
    let pointer = buffer.contents().bindMemory(to: T.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func makeBuffer<T>(_ device: MTLDevice, values: [T]) -> MTLBuffer {
    let byteCount = values.count * MemoryLayout<T>.stride
    return values.withUnsafeBytes { bytes in
        device.makeBuffer(bytes: bytes.baseAddress!, length: byteCount, options: .storageModeShared)!
    }
}

func dispatch(
    queue: MTLCommandQueue,
    pipeline: MTLComputePipelineState,
    input: MTLBuffer,
    output: MTLBuffer,
    params: MTLBuffer,
    threadCount: Int
) -> (gpuNS: UInt64, wallNS: UInt64) {
    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.setBuffer(params, offset: 0, index: 2)
    let width = min(256, pipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(
        MTLSize(width: threadCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
    let wallStart = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds
    guard commandBuffer.status == .completed else {
        fatalError("Metal command failed: \(commandBuffer.error?.localizedDescription ?? "unknown")")
    }
    let gpuNS = UInt64((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000_000_000.0)
    return (gpuNS, wallEnd - wallStart)
}

func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
    let index = Int(floor(Double(sorted.count - 1) * fraction))
    return sorted[index]
}

func evenPopulationMedian(_ sorted: [Double]) -> Double {
    precondition(!sorted.isEmpty)
    let upper = sorted.count / 2
    if sorted.count % 2 == 1 {
        return sorted[upper]
    }
    return (sorted[upper - 1] + sorted[upper]) / 2.0
}

guard CommandLine.arguments.count == 2 else {
    fatalError("usage: metal-mulhi-probe /path/to/probe.metallib")
}
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal unavailable")
}
let queue = device.makeCommandQueue()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))

func pipeline(_ name: String) -> MTLComputePipelineState {
    try! device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}

let checkMulhi = pipeline("check_mulhi")
let checkWidened = pipeline("check_widened")
let benchMulhi = pipeline("bench_mulhi")
let benchWidened = pipeline("bench_widened")

print("device=\(device.name)")
print("registry_id=\(device.registryID)")
print("thread_execution_width_mulhi=\(benchMulhi.threadExecutionWidth)")
print("thread_execution_width_widened=\(benchWidened.threadExecutionWidth)")

var pairs: [SIMD2<UInt32>] = []
let boundaries: [UInt32] = [0, 1, 2, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff]
for a in boundaries {
    for b in boundaries {
        pairs.append(SIMD2(a, b))
    }
}
var rng = SplitMix64(state: 0x6a09e667f3bcc909)
while pairs.count < 16_384 {
    pairs.append(SIMD2(UInt32(truncatingIfNeeded: rng.next()), UInt32(truncatingIfNeeded: rng.next())))
}
let pairInput = makeBuffer(device, values: pairs)
let pairOutputMulhi = device.makeBuffer(length: pairs.count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
let pairOutputWidened = device.makeBuffer(length: pairs.count * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
var pairCount = UInt32(pairs.count)
let pairParams = device.makeBuffer(bytes: &pairCount, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
_ = dispatch(queue: queue, pipeline: checkMulhi, input: pairInput, output: pairOutputMulhi, params: pairParams, threadCount: pairs.count)
_ = dispatch(queue: queue, pipeline: checkWidened, input: pairInput, output: pairOutputWidened, params: pairParams, threadCount: pairs.count)
let checkedMulhi = bufferArray(pairOutputMulhi, count: pairs.count, as: SIMD2<UInt32>.self)
let checkedWidened = bufferArray(pairOutputWidened, count: pairs.count, as: SIMD2<UInt32>.self)
for index in pairs.indices {
    let product = UInt64(pairs[index].x) * UInt64(pairs[index].y)
    let expected = SIMD2(UInt32(truncatingIfNeeded: product), UInt32(truncatingIfNeeded: product >> 32))
    guard checkedMulhi[index] == expected, checkedWidened[index] == expected else {
        fatalError("product mismatch at \(index): input=\(pairs[index]), mulhi=\(checkedMulhi[index]), widened=\(checkedWidened[index]), expected=\(expected)")
    }
}
print("product_correctness_cases=\(pairs.count)")

let correctnessThreads = 257
let correctnessIterations: UInt32 = 19
var correctnessSeeds: [UInt32] = []
for _ in 0..<correctnessThreads {
    correctnessSeeds.append(UInt32(truncatingIfNeeded: rng.next()))
}
let correctnessInput = makeBuffer(device, values: correctnessSeeds)
let correctnessMulhi = device.makeBuffer(length: correctnessSeeds.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
let correctnessWidened = device.makeBuffer(length: correctnessSeeds.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
var correctnessParamsValue = BenchParams(threadCount: UInt32(correctnessThreads), iterations: correctnessIterations)
let correctnessParams = device.makeBuffer(bytes: &correctnessParamsValue, length: MemoryLayout<BenchParams>.stride, options: .storageModeShared)!
_ = dispatch(queue: queue, pipeline: benchMulhi, input: correctnessInput, output: correctnessMulhi, params: correctnessParams, threadCount: correctnessThreads)
_ = dispatch(queue: queue, pipeline: benchWidened, input: correctnessInput, output: correctnessWidened, params: correctnessParams, threadCount: correctnessThreads)
let correctnessMulhiValues = bufferArray(correctnessMulhi, count: correctnessThreads, as: UInt32.self)
let correctnessWidenedValues = bufferArray(correctnessWidened, count: correctnessThreads, as: UInt32.self)
for index in 0..<correctnessThreads {
    var expected = correctnessSeeds[index]
    for _ in 0..<correctnessIterations {
        expected = cpuStep(expected)
    }
    guard correctnessMulhiValues[index] == expected, correctnessWidenedValues[index] == expected else {
        fatalError("chain mismatch at \(index)")
    }
}
print("chain_correctness_threads=\(correctnessThreads)")
print("chain_correctness_iterations=\(correctnessIterations)")

let benchmarkThreads = 131_072
let benchmarkIterations: UInt32 = 65_536
let samples = 20
var benchmarkSeeds: [UInt32] = []
benchmarkSeeds.reserveCapacity(benchmarkThreads)
for _ in 0..<benchmarkThreads {
    benchmarkSeeds.append(UInt32(truncatingIfNeeded: rng.next()))
}
let benchmarkInput = makeBuffer(device, values: benchmarkSeeds)
let benchmarkOutputMulhi = device.makeBuffer(length: benchmarkSeeds.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
let benchmarkOutputWidened = device.makeBuffer(length: benchmarkSeeds.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
var benchmarkParamsValue = BenchParams(threadCount: UInt32(benchmarkThreads), iterations: benchmarkIterations)
let benchmarkParams = device.makeBuffer(bytes: &benchmarkParamsValue, length: MemoryLayout<BenchParams>.stride, options: .storageModeShared)!

for _ in 0..<3 {
    _ = dispatch(queue: queue, pipeline: benchMulhi, input: benchmarkInput, output: benchmarkOutputMulhi, params: benchmarkParams, threadCount: benchmarkThreads)
    _ = dispatch(queue: queue, pipeline: benchWidened, input: benchmarkInput, output: benchmarkOutputWidened, params: benchmarkParams, threadCount: benchmarkThreads)
}

var mulhiGPU: [UInt64] = []
var widenedGPU: [UInt64] = []
var mulhiWall: [UInt64] = []
var widenedWall: [UInt64] = []
for sample in 0..<samples {
    let firstMulhi = sample % 2 == 0
    let first = dispatch(
        queue: queue,
        pipeline: firstMulhi ? benchMulhi : benchWidened,
        input: benchmarkInput,
        output: firstMulhi ? benchmarkOutputMulhi : benchmarkOutputWidened,
        params: benchmarkParams,
        threadCount: benchmarkThreads)
    let second = dispatch(
        queue: queue,
        pipeline: firstMulhi ? benchWidened : benchMulhi,
        input: benchmarkInput,
        output: firstMulhi ? benchmarkOutputWidened : benchmarkOutputMulhi,
        params: benchmarkParams,
        threadCount: benchmarkThreads)
    let m = firstMulhi ? first : second
    let w = firstMulhi ? second : first
    mulhiGPU.append(m.gpuNS)
    widenedGPU.append(w.gpuNS)
    mulhiWall.append(m.wallNS)
    widenedWall.append(w.wallNS)
    print("sample=\(sample) order=\(firstMulhi ? "mulhi,widened" : "widened,mulhi") mulhi_gpu_ns=\(m.gpuNS) widened_gpu_ns=\(w.gpuNS) mulhi_wall_ns=\(m.wallNS) widened_wall_ns=\(w.wallNS)")
}

let finalMulhi = bufferArray(benchmarkOutputMulhi, count: benchmarkThreads, as: UInt32.self)
let finalWidened = bufferArray(benchmarkOutputWidened, count: benchmarkThreads, as: UInt32.self)
guard finalMulhi == finalWidened else {
    fatalError("observable benchmark outputs differ")
}
var checksum: UInt64 = 0
for value in finalMulhi {
    checksum &+= UInt64(value)
}
print("benchmark_output_checksum=\(checksum)")

let completeProducts = Double(benchmarkThreads) * Double(benchmarkIterations)
func summarize(_ label: String, _ values: [UInt64]) {
    let rates = values.map { completeProducts / (Double($0) / 1_000_000_000.0) }
    let sortedTimes = values.map { Double($0) }.sorted()
    let sortedRates = rates.sorted()
    let medianTime = evenPopulationMedian(sortedTimes)
    let medianRate = evenPopulationMedian(sortedRates)
    let p5Rate = percentile(sortedRates, 0.05)
    print("\(label)_median_gpu_ns=\(medianTime)")
    print("\(label)_median_complete_products_per_s=\(medianRate)")
    print("\(label)_p5_complete_products_per_s=\(p5Rate)")
}
print("benchmark_threads=\(benchmarkThreads)")
print("benchmark_iterations=\(benchmarkIterations)")
print("complete_products_per_sample=\(UInt64(completeProducts))")
summarize("mulhi", mulhiGPU)
summarize("widened", widenedGPU)
let pairedRatios = zip(mulhiGPU, widenedGPU).map { Double($0.1) / Double($0.0) }
let sortedRatios = pairedRatios.sorted()
print("paired_mulhi_speedup_median=\(evenPopulationMedian(sortedRatios))")
print("paired_mulhi_speedup_min=\(sortedRatios.first!)")
print("paired_mulhi_speedup_max=\(sortedRatios.last!)")
