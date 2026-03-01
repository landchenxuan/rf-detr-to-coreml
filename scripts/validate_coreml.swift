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

// Cleanup
try? FileManager.default.removeItem(at: compiledURL)

print()
print("=== Validation Complete ===")
