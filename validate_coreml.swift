#!/usr/bin/env swift
// Validate our CoreML model using Apple's native ML framework.
// This simulates how a real iOS/macOS app (including Roboflow Swift SDK) would load and run the model.

import Foundation
import CoreML
import CoreImage
import Vision

let modelPath = "output/rf-detr-base-fp32.mlpackage"

print("=== CoreML Model Validation ===")
print()

// Step 1: Compile the model (same as MLModel.compileModel on iOS)
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
    ("ALL (GPU + Neural Engine)", .all),
    ("CPU_AND_NE (Roboflow config)", .cpuAndNeuralEngine),
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

// Step 3: Inspect model spec
print()
print("3. Model specification:")
if let (_, model) = models.first {
    let desc = model.modelDescription

    print("   Inputs:")
    for (name, feat) in desc.inputDescriptionsByName {
        print("     \(name): \(feat.type.rawValue), constraint=\(feat.imageConstraint?.pixelsWide ?? 0)x\(feat.imageConstraint?.pixelsHigh ?? 0)")
        if let imgConstraint = feat.imageConstraint {
            print("       pixelFormat: \(imgConstraint.pixelFormatType)")
            print("       size: \(imgConstraint.pixelsWide)x\(imgConstraint.pixelsHigh)")
        }
        if let multiConstraint = feat.multiArrayConstraint {
            print("       shape: \(multiConstraint.shape)")
            print("       dataType: \(multiConstraint.dataType.rawValue)")
        }
    }

    print("   Outputs:")
    for (name, feat) in desc.outputDescriptionsByName {
        print("     \(name): type=\(feat.type.rawValue)")
        if let multiConstraint = feat.multiArrayConstraint {
            print("       shape: \(multiConstraint.shape)")
            print("       dataType: \(multiConstraint.dataType.rawValue)")
        }
    }

    print("   Metadata:")
    print("     author: \(desc.metadata[.author] ?? "n/a")")
    print("     description: \(desc.metadata[.description] ?? "n/a")")
    print("     version: \(desc.metadata[.versionString] ?? "n/a")")
}

// Step 4: Create test input (560x560 random image)
print()
print("4. Running inference with test image...")
let resolution = 560
let pixelCount = resolution * resolution
var pixelData = [UInt8](repeating: 0, count: pixelCount * 4) // RGBA
for i in 0..<pixelCount {
    pixelData[i * 4 + 0] = UInt8.random(in: 0...255) // R
    pixelData[i * 4 + 1] = UInt8.random(in: 0...255) // G
    pixelData[i * 4 + 2] = UInt8.random(in: 0...255) // B
    pixelData[i * 4 + 3] = 255 // A
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
print("   Test image: \(resolution)x\(resolution) RGBA")

// Step 5: Run inference on all compute unit configs and compare
var allOutputs: [(String, [String: MLMultiArray])] = []

for (name, model) in models {
    let startTime = CFAbsoluteTimeGetCurrent()

    // Use VNCoreMLRequest like Roboflow SDK does
    guard let visionModel = try? VNCoreMLModel(for: model) else {
        print("   FAIL: \(name) — could not create VNCoreMLModel")
        continue
    }

    let semaphore = DispatchSemaphore(value: 0)
    var outputArrays: [String: MLMultiArray] = [:]
    var inferenceError: Error? = nil

    let request = VNCoreMLRequest(model: visionModel) { request, error in
        defer { semaphore.signal() }
        if let error = error {
            inferenceError = error
            return
        }
        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
            inferenceError = NSError(domain: "test", code: 1, userInfo: [NSLocalizedDescriptionKey: "No results"])
            return
        }
        for result in results {
            if let arr = result.featureValue.multiArrayValue {
                outputArrays[result.featureName] = arr
            }
        }
    }
    request.imageCropAndScaleOption = .scaleFill

    let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    do {
        try handler.perform([request])
    } catch {
        print("   FAIL: \(name) — \(error)")
        continue
    }
    semaphore.wait()

    let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000

    if let error = inferenceError {
        print("   FAIL: \(name) — \(error)")
        continue
    }

    print("   OK: \(name) — \(elapsed.rounded())ms, outputs: \(outputArrays.keys.sorted())")
    for (key, arr) in outputArrays.sorted(by: { $0.key < $1.key }) {
        print("       \(key): shape=\(arr.shape), type=\(arr.dataType.rawValue)")
    }

    allOutputs.append((name, outputArrays))
}

// Step 6: Cross-validate outputs between compute units
print()
print("5. Cross-validation between compute units:")
if allOutputs.count >= 2 {
    let (name0, out0) = allOutputs[0]
    for i in 1..<allOutputs.count {
        let (nameI, outI) = allOutputs[i]

        for key in out0.keys.sorted() {
            guard let arr0 = out0[key], let arrI = outI[key] else {
                print("   MISSING: \(key) in \(nameI)")
                continue
            }

            let count = arr0.count
            var maxDiff: Float = 0
            var sumDiff: Float = 0

            let ptr0 = arr0.dataPointer.assumingMemoryBound(to: Float.self)
            let ptrI = arrI.dataPointer.assumingMemoryBound(to: Float.self)

            for j in 0..<count {
                let diff = abs(ptr0[j] - ptrI[j])
                maxDiff = max(maxDiff, diff)
                sumDiff += diff
            }
            let avgDiff = sumDiff / Float(count)

            print("   \(name0) vs \(nameI)")
            print("     \(key): maxDiff=\(maxDiff), avgDiff=\(avgDiff)")
        }
    }
}

// Step 7: Latency benchmark
print()
print("6. Latency benchmark (20 runs each):")
for (name, model) in models {
    guard let visionModel = try? VNCoreMLModel(for: model) else { continue }

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

// Cleanup compiled model
try? FileManager.default.removeItem(at: compiledURL)

print()
print("=== Validation Complete ===")
