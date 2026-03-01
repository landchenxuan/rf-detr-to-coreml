#!/usr/bin/env swift
//
// Validate a CoreML model using Apple's native ML framework.
// Simulates how a real iOS/macOS app loads and runs the model.
//
// Usage:
//   swift scripts/validate_coreml.swift output/rf-detr-nano-fp32.mlpackage
//   swift scripts/validate_coreml.swift output/rf-detr-seg-nano-fp32.mlpackage

import Foundation
import CoreML
import CoreImage
import Vision

// Parse command line
guard CommandLine.arguments.count >= 2 else {
    print("Usage: swift \(CommandLine.arguments[0]) <path-to-mlpackage>")
    exit(1)
}
let modelPath = CommandLine.arguments[1]

print("=== CoreML Model Validation ===")
print("Model: \(modelPath)")
print()

// Step 1: Compile the model
print("1. Compiling model...")
let modelURL = URL(fileURLWithPath: modelPath)
let compiledURL: URL
do {
    compiledURL = try MLModel.compileModel(at: modelURL)
    print("   OK: Compiled to \(compiledURL.lastPathComponent)")
} catch {
    print("   FAIL: \(error)")
    exit(1)
}

// Step 2: Load with different compute unit configurations
let configs: [(String, MLComputeUnits)] = [
    ("ALL (CPU+GPU)", .all),
    ("CPU_AND_NE", .cpuAndNeuralEngine),
    ("CPU_ONLY", .cpuOnly),
]

var models: [(String, MLModel)] = []
print()
print("2. Loading model with different compute units...")
for (name, units) in configs {
    let config = MLModelConfiguration()
    config.computeUnits = units
    do {
        let model = try MLModel(contentsOf: compiledURL, configuration: config)
        models.append((name, model))
        print("   OK: \(name)")
    } catch {
        print("   FAIL: \(name) — \(error)")
    }
}

// Step 3: Inspect model spec (auto-detect resolution)
print()
print("3. Model specification:")
var resolution = 0
if let (_, model) = models.first {
    let desc = model.modelDescription

    print("   Inputs:")
    for (name, feat) in desc.inputDescriptionsByName {
        if let imgConstraint = feat.imageConstraint {
            resolution = imgConstraint.pixelsWide
            print("     \(name): Image \(imgConstraint.pixelsWide)x\(imgConstraint.pixelsHigh)")
        } else if let multiConstraint = feat.multiArrayConstraint {
            print("     \(name): MultiArray shape=\(multiConstraint.shape)")
        }
    }

    print("   Outputs:")
    for (name, feat) in desc.outputDescriptionsByName {
        if let multiConstraint = feat.multiArrayConstraint {
            print("     \(name): shape=\(multiConstraint.shape), dtype=\(multiConstraint.dataType.rawValue)")
        } else {
            print("     \(name): type=\(feat.type.rawValue)")
        }
    }
}

guard resolution > 0 else {
    print("   FAIL: Could not detect input resolution from model spec")
    exit(1)
}

// Step 4: Create test image
print()
print("4. Running inference with \(resolution)x\(resolution) test image...")
let pixelCount = resolution * resolution
var pixelData = [UInt8](repeating: 0, count: pixelCount * 4)
for i in 0..<pixelCount {
    pixelData[i * 4 + 0] = UInt8.random(in: 0...255)
    pixelData[i * 4 + 1] = UInt8.random(in: 0...255)
    pixelData[i * 4 + 2] = UInt8.random(in: 0...255)
    pixelData[i * 4 + 3] = 255
}

let colorSpace = CGColorSpaceCreateDeviceRGB()
let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
guard let context = CGContext(
    data: &pixelData,
    width: resolution,
    height: resolution,
    bitsPerComponent: 8,
    bytesPerRow: resolution * 4,
    space: colorSpace,
    bitmapInfo: bitmapInfo.rawValue
), let cgImage = context.makeImage() else {
    print("   FAIL: Could not create test image")
    exit(1)
}
let ciImage = CIImage(cgImage: cgImage)

// Step 5: Run inference and benchmark
print()
print("5. Latency benchmark (20 runs each):")
for (name, model) in models {
    guard let visionModel = try? VNCoreMLModel(for: model) else {
        print("   FAIL: \(name) — could not create VNCoreMLModel")
        continue
    }

    // Warmup
    for _ in 0..<3 {
        let req = VNCoreMLRequest(model: visionModel) { _, _ in }
        req.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(ciImage: ciImage, options: [:]).perform([req])
    }

    // Timed runs
    var times: [Double] = []
    for _ in 0..<20 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let req = VNCoreMLRequest(model: visionModel) { _, _ in }
        req.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(ciImage: ciImage, options: [:]).perform([req])
        times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
    }

    times.sort()
    let median = times[times.count / 2]
    let p5 = times[max(0, times.count / 20)]
    let p95 = times[min(times.count - 1, times.count * 19 / 20)]
    print("   \(name): median=\(String(format: "%.1f", median))ms, P5-P95=[\(String(format: "%.1f", p5)), \(String(format: "%.1f", p95))]")
}

// Step 6: MLComputePlan analysis (macOS 14.4+ / iOS 17.4+)
print()
print("6. MLComputePlan analysis (per-op device assignment)...")

if #available(macOS 14.4, iOS 17.4, *) {
    let planConfig = MLModelConfiguration()
    planConfig.computeUnits = .all

    // MLComputePlan.load requires a compiled model URL and uses async/await
    let semaphore = DispatchSemaphore(value: 0)
    var cpuGpuCapable = 0
    var aneCapable = 0
    var noDevice = 0
    var totalOps = 0
    var planFailed = false

    Task {
        do {
            let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: planConfig)

            // modelStructure is an enum — pattern match for .program
            switch plan.modelStructure {
            case .program(let program):
                for (_, function) in program.functions {
                    // Function has a single .block (not .blocks)
                    for operation in function.block.operations {
                        totalOps += 1
                        if let usage = plan.deviceUsage(for: operation) {
                            let supported = usage.supported
                            let hasANE = supported.contains { device in
                                if case .neuralEngine = device { return true }
                                return false
                            }
                            let hasCPUorGPU = supported.contains { device in
                                switch device {
                                case .cpu, .gpu: return true
                                default: return false
                                }
                            }

                            if hasANE {
                                aneCapable += 1
                            } else if hasCPUorGPU {
                                cpuGpuCapable += 1
                            } else {
                                noDevice += 1
                            }
                        } else {
                            noDevice += 1
                        }
                    }
                }
            default:
                print("   SKIP: Model is not an MLProgram (NeuralNetwork or Pipeline)")
            }
        } catch {
            print("   FAIL: Could not load compute plan — \(error)")
            planFailed = true
        }
        semaphore.signal()
    }
    semaphore.wait()

    if !planFailed && totalOps > 0 {
        let computeOps = cpuGpuCapable + aneCapable
        let cpuGpuPct = computeOps > 0 ? String(format: "%.0f%%", Double(cpuGpuCapable) / Double(computeOps) * 100) : "—"
        let anePct = computeOps > 0 ? String(format: "%.0f%%", Double(aneCapable) / Double(computeOps) * 100) : "—"

        print("   CPU+GPU capable: \(cpuGpuCapable) (\(cpuGpuPct))")
        print("   Neural Engine capable: \(aneCapable) (\(anePct))")
        print("   No device (const/reshape): \(noDevice)")
        print("   Total ops: \(totalOps)")

        if aneCapable == 0 {
            print()
            print("   Note: Zero ops are Neural Engine capable.")
            print("   Root cause: FP32 precision — ANE only operates in FP16.")
        }
    }
} else {
    print("   SKIP: MLComputePlan requires macOS 14.4+ / iOS 17.4+")
}

// Cleanup
try? FileManager.default.removeItem(at: compiledURL)

print()
print("=== Validation Complete ===")
