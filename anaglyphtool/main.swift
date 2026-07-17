//
//  main.swift
//  anaglyphtool
//
//  Created by Scott Jann on 12/16/25.
//

import Foundation
import CoreImage
import AppKit
import Metal
import ImageIO

// MARK: - Anaglyph Filter using Metal Kernels

enum AnaglyphFilter {
	enum Mode: String {
		case simple           // Simple R/GB separation
		case optimized        // Optimized with better color
		case dubois           // Dubois method for color preservation
		case grayscale        // Grayscale anaglyph

		var kernelString: String {
			switch self {
			case .simple:
				return """
				#include <CoreImage/CoreImage.h>
				using namespace metal;
				[[ stitchable ]] float4 anaglyph_simple(coreimage::sample_t leftImage, coreimage::sample_t rightImage) {
					return float4(leftImage.r, rightImage.g, rightImage.b, 1.0);
				}
				"""

			case .optimized:
				return """
				#include <CoreImage/CoreImage.h>
				using namespace metal;
				[[ stitchable ]] float4 anaglyph_optimized(coreimage::sample_t leftImage, coreimage::sample_t rightImage) {
					// Optimized matrices for better depth perception
					float r = leftImage.r * 0.4561 + leftImage.g * 0.500484 + leftImage.b * 0.176381;
					float g = rightImage.r * 0.378476 + rightImage.g * 0.73364 + rightImage.b * 0.0184559;
					float b = rightImage.r * -0.0261502 + rightImage.g * -0.0736177 + rightImage.b * 1.22684;
					return float4(r, g, b, 1.0);
				}
				"""

			case .dubois:
				return """
				#include <CoreImage/CoreImage.h>
				using namespace metal;
				[[ stitchable ]] float4 anaglyph_dubois(coreimage::sample_t leftImage, coreimage::sample_t rightImage) {
					// Dubois method for better color preservation
					float r = leftImage.r * 0.437 + leftImage.g * 0.449 + leftImage.b * 0.164
							+ rightImage.r * -0.011 + rightImage.g * -0.032 + rightImage.b * -0.007;
					float g = leftImage.r * -0.062 + leftImage.g * -0.062 + leftImage.b * -0.024
							+ rightImage.r * 0.377 + rightImage.g * 0.761 + rightImage.b * 0.009;
					float b = leftImage.r * -0.048 + leftImage.g * -0.050 + leftImage.b * -0.017
							+ rightImage.r * -0.026 + rightImage.g * -0.093 + rightImage.b * 1.234;
					return float4(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
				}
				"""

			case .grayscale:
				return """
				#include <CoreImage/CoreImage.h>
				using namespace metal;
				[[ stitchable ]] float4 anaglyph_grayscale(coreimage::sample_t leftImage, coreimage::sample_t rightImage) {
					// Convert to grayscale using luminance weights
					float leftGray = leftImage.r * 0.299 + leftImage.g * 0.587 + leftImage.b * 0.114;
					float rightGray = rightImage.r * 0.299 + rightImage.g * 0.587 + rightImage.b * 0.114;
					return float4(leftGray, rightGray, rightGray, 1.0);
				}
				"""
			}
		}

		var fileSuffix: String {
			switch self {
			case .simple:
				return "anaglyph"
			case .optimized:
				return "anaglyph-opt"
			case .dubois:
				return "anaglyph-dubois"
			case .grayscale:
				return "anaglyph-gray"
			}
		}
	}

	// Compiled kernels are cached so batch runs don't recompile per image
	private static var kernelCache: [Mode: CIKernel] = [:]
	private static let kernelLock = NSLock()

	static func kernel(for mode: Mode) throws -> CIKernel {
		kernelLock.lock()
		defer { kernelLock.unlock() }

		if let cached = kernelCache[mode] {
			return cached
		}

		guard let kernel = try CIColorKernel.kernels(withMetalString: mode.kernelString).first else {
			throw AnaglyphError.failedToCompileKernel(mode.rawValue)
		}

		kernelCache[mode] = kernel
		return kernel
	}

	static func apply(_ mode: Mode, left: CIImage, right: CIImage) throws -> CIImage {
		let kernel = try kernel(for: mode)

		guard let output = kernel.apply(
			extent: left.extent,
			roiCallback: { _, rect in rect },
			arguments: [left, right]
		) else {
			throw AnaglyphError.failedToGenerateOutput
		}

		return output
	}
}

// MARK: - Stereo Disparity Analyzer

class StereoDisparityAnalyzer {

	struct DisparityResult {
		let suggestedOffset: Int
		let nearDisparity: Float   // Disparity of closest objects
		let farDisparity: Float    // Disparity of furthest objects
		let mainSubjectDisparity: Float  // Estimated main subject
		let confidence: Float
	}

	// Grayscale pixels rendered once per image, safe to read from any thread
	private struct GrayscaleBuffer {
		let pixels: [UInt8]
		let width: Int
		let height: Int

		subscript(x: Int, y: Int) -> Float {
			return Float(pixels[y * width + x])
		}
	}

	// Analyze stereo pair to find optimal convergence
	static func analyzeDisparity(
		left: CIImage,
		right: CIImage,
		context: CIContext,
		verbose: Bool = false,
		fast: Bool = false
	) -> DisparityResult {

		// Adjust grid size for fast mode
		let gridSize = fast ? 40 : 20  // Larger grid = fewer samples = faster

		// Render both images to grayscale pixel buffers once, up front
		let startRender = Date()
		guard let leftBuffer = renderGrayscale(left, context: context),
			  let rightBuffer = renderGrayscale(right, context: context) else {
			return DisparityResult(
				suggestedOffset: 0,
				nearDisparity: 0,
				farDisparity: 0,
				mainSubjectDisparity: 0,
				confidence: 0
			)
		}

		if verbose {
			print("    Image rendering: \(String(format: "%.2f", Date().timeIntervalSince(startRender)))s")
		}

		// Find feature points and their disparities
		let disparities = findDisparities(
			left: leftBuffer,
			right: rightBuffer,
			gridSize: gridSize,
			verbose: verbose
		)

		guard !disparities.isEmpty else {
			return DisparityResult(
				suggestedOffset: 0,
				nearDisparity: 0,
				farDisparity: 0,
				mainSubjectDisparity: 0,
				confidence: 0
			)
		}

		// Sort disparities to find range
		let sorted = disparities.sorted()
		let nearDisparity = sorted.last ?? 0
		let farDisparity = sorted.first ?? 0

		// Find main subject disparity (using several heuristics)
		let mainSubjectDisparity = findMainSubjectDisparity(disparities: disparities)

		// Calculate optimal offset
		// We want to set the convergence so the main subject appears at screen depth (zero disparity)
		// offset = -mainSubjectDisparity puts the main subject at screen depth
		let suggestedOffset = Int(-mainSubjectDisparity)

		// Calculate confidence based on disparity distribution
		let confidence = calculateConfidence(disparities: disparities)

		if verbose {
			print("\n  Disparity Analysis:")
			print("    Samples analyzed: \(disparities.count)")
			print("    Near objects: \(Int(nearDisparity)) pixels disparity")
			print("    Far objects: \(Int(farDisparity)) pixels disparity")
			print("    Main subject: \(Int(mainSubjectDisparity)) pixels disparity")
			print("    Suggested offset: \(suggestedOffset) pixels")
			print("    Confidence: \(String(format: "%.1f%%", confidence * 100))")
		}

		return DisparityResult(
			suggestedOffset: suggestedOffset,
			nearDisparity: nearDisparity,
			farDisparity: farDisparity,
			mainSubjectDisparity: mainSubjectDisparity,
			confidence: confidence
		)
	}

	// Render a CIImage into a single-channel 8-bit grayscale buffer
	private static func renderGrayscale(_ image: CIImage, context: CIContext) -> GrayscaleBuffer? {
		let width = Int(image.extent.width)
		let height = Int(image.extent.height)

		guard width > 0, height > 0 else {
			return nil
		}

		var rgba = [UInt8](repeating: 0, count: width * height * 4)
		rgba.withUnsafeMutableBytes { buffer in
			context.render(
				image,
				toBitmap: buffer.baseAddress!,
				rowBytes: width * 4,
				bounds: image.extent,
				format: .RGBA8,
				colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
			)
		}

		var pixels = [UInt8](repeating: 0, count: width * height)
		for i in 0..<(width * height) {
			let r = Float(rgba[i * 4])
			let g = Float(rgba[i * 4 + 1])
			let b = Float(rgba[i * 4 + 2])
			pixels[i] = UInt8(min(r * 0.299 + g * 0.587 + b * 0.114, 255))
		}

		return GrayscaleBuffer(pixels: pixels, width: width, height: height)
	}

	// Find disparities using block matching (parallelized)
	private static func findDisparities(
		left: GrayscaleBuffer,
		right: GrayscaleBuffer,
		gridSize: Int,
		verbose: Bool
	) -> [Float] {

		let blockSize = 16  // Size of matching block
		let searchRange = left.width / 10  // Max 10% of width for speed

		// Focus on center 80% to avoid edge artifacts
		let startX = left.width / 10
		let endX = left.width * 9 / 10
		let startY = left.height / 10
		let endY = left.height * 9 / 10

		// Sample points in a grid pattern
		var samplePoints: [(x: Int, y: Int)] = []
		for y in stride(from: startY, to: endY, by: gridSize) {
			for x in stride(from: startX, to: endX, by: gridSize) {
				samplePoints.append((x: x, y: y))
			}
		}

		if verbose {
			print("    Analyzing \(samplePoints.count) sample points...")
		}

		let startAnalysis = Date()

		// Process points in parallel; each iteration writes only to its own slot
		var results = [Float?](repeating: nil, count: samplePoints.count)
		results.withUnsafeMutableBufferPointer { buffer in
			DispatchQueue.concurrentPerform(iterations: samplePoints.count) { i in
				buffer[i] = findPointDisparity(
					point: samplePoints[i],
					left: left,
					right: right,
					blockSize: blockSize,
					searchRange: searchRange
				)
			}
		}

		if verbose {
			print("    Block matching: \(String(format: "%.2f", Date().timeIntervalSince(startAnalysis)))s")
		}

		return results.compactMap { $0 }
	}

	// Find disparity for a single point using block matching
	private static func findPointDisparity(
		point: (x: Int, y: Int),
		left: GrayscaleBuffer,
		right: GrayscaleBuffer,
		blockSize: Int,
		searchRange: Int
	) -> Float? {

		// Check bounds
		let halfBlock = blockSize / 2
		guard left.width == right.width,
			  left.height == right.height,
			  point.x >= halfBlock,
			  point.x < left.width - halfBlock,
			  point.y >= halfBlock,
			  point.y < left.height - halfBlock else {
			return nil
		}

		var bestDisparity: Float = 0
		var bestScore = Float.infinity

		// Search along the epipolar line (same y-coordinate)
		for xOffset in stride(from: 0, through: searchRange, by: 2) {
			var sum: Float = 0
			var count: Float = 0

			// Compare blocks
			for dy in -halfBlock..<halfBlock {
				let y = point.y + dy
				for dx in -halfBlock..<halfBlock {
					let leftX = point.x + dx
					let rightX = leftX - xOffset

					guard rightX >= 0 else { continue }

					sum += abs(left[leftX, y] - right[rightX, y])
					count += 1
				}
			}

			if count > 0 {
				let score = sum / count
				if score < bestScore {
					bestScore = score
					bestDisparity = Float(xOffset)
				}
			}
		}

		// Only return if we have a confident match (low average difference)
		return bestScore < 50 ? bestDisparity : nil
	}

	// Find the main subject disparity using heuristics
	private static func findMainSubjectDisparity(disparities: [Float]) -> Float {

		guard !disparities.isEmpty else { return 0 }

		// Strategy 1: Use the median of the middle 50% of disparities
		// This assumes the main subject occupies the middle depth range
		let sorted = disparities.sorted()
		let q1Index = sorted.count / 4
		let q3Index = sorted.count * 3 / 4
		let middleRange = Array(sorted[q1Index..<q3Index])

		if !middleRange.isEmpty {
			// Use median of middle range
			return middleRange[middleRange.count / 2]
		}

		// Fallback: use overall median
		return sorted[sorted.count / 2]
	}

	// Calculate confidence based on disparity distribution
	private static func calculateConfidence(disparities: [Float]) -> Float {
		guard disparities.count > 10 else { return 0 }

		// Calculate standard deviation
		let mean = disparities.reduce(0, +) / Float(disparities.count)
		let variance = disparities.map { pow($0 - mean, 2) }.reduce(0, +) / Float(disparities.count)
		let stdDev = sqrt(variance)

		// Lower standard deviation = more consistent disparities = higher confidence
		// Normalize to 0-1 range
		let normalizedStdDev = min(stdDev / 50.0, 1.0)  // 50 pixels as max expected stddev
		let confidence = 1.0 - normalizedStdDev

		// Boost confidence if we have many samples
		let sampleBoost = min(Float(disparities.count) / 100.0, 1.0)

		return confidence * 0.7 + sampleBoost * 0.3
	}
}

// MARK: - Anaglyph Converter

class AnaglyphConverter {
	let ciContext: CIContext
	let verbose: Bool

	init(verbose: Bool = false) {
		self.verbose = verbose

		// Create Metal-backed context
		guard let metalDevice = MTLCreateSystemDefaultDevice() else {
			fatalError("Metal is not supported on this device")
		}

		self.ciContext = CIContext(mtlDevice: metalDevice)

		if verbose {
			print("Initialized with Metal device: \(metalDevice.name)")
		}
	}

	func processImage(
		at inputPath: String,
		outputDirectory: String? = nil,
		mode: AnaglyphFilter.Mode = .simple,
		quality: Float = 0.9,
		autoDetect: Bool = false,
		fastMode: Bool = false,
		useModeNaming: Bool = false,
		manualOffset: Int? = nil
	) throws {
		if verbose {
			print("\nProcessing: \(inputPath)")
			print("  Mode: \(mode.rawValue)")
		}

		let startTime = Date()

		// Load the image, applying any EXIF orientation
		let inputURL = URL(fileURLWithPath: inputPath)
		guard var ciImage = CIImage(contentsOf: inputURL, options: [.applyOrientationProperty: true]) else {
			throw AnaglyphError.failedToLoadImage(inputPath)
		}

		// Normalize the origin so the cropping math below can assume (0, 0)
		if ciImage.extent.origin != .zero {
			ciImage = ciImage.transformed(by: CGAffineTransform(
				translationX: -ciImage.extent.origin.x,
				y: -ciImage.extent.origin.y
			))
		}

		// Get dimensions and split the side-by-side image
		let extent = ciImage.extent
		let width = extent.width
		let height = extent.height
		let halfWidth = width / 2

		if verbose {
			print("  Input dimensions: \(Int(width)) x \(Int(height))")
			print("  Each eye: \(Int(halfWidth)) x \(Int(height))")
		}

		// Extract left and right images
		let leftRect = CGRect(x: 0, y: 0, width: halfWidth, height: height)
		let rightRect = CGRect(x: halfWidth, y: 0, width: halfWidth, height: height)

		var leftImage = ciImage.cropped(to: leftRect)
		var rightImage = ciImage.cropped(to: rightRect)

		// Move right image to same position as left
		rightImage = rightImage.transformed(by: CGAffineTransform(translationX: -halfWidth, y: 0))

		// Determine offset
		let offset: Int
		if let manual = manualOffset {
			offset = manual
			if verbose {
				print("  Using manual offset: \(offset) pixels")
			}
		} else if autoDetect {
			if verbose {
				print("  Analyzing stereo disparity\(fastMode ? " (fast mode)" : "")...")
			}

			let disparityResult = StereoDisparityAnalyzer.analyzeDisparity(
				left: leftImage,
				right: rightImage,
				context: ciContext,
				verbose: verbose,
				fast: fastMode
			)

			offset = disparityResult.suggestedOffset

			print("  ✓ Auto-detected offset: \(offset) pixels (confidence: \(String(format: "%.0f%%", disparityResult.confidence * 100)))")
		} else {
			offset = 0
			if verbose {
				print("  No offset applied")
			}
		}

		// Apply offset if specified
		if offset != 0 {
			// Split the offset between both images for centered convergence
			let halfOffset = CGFloat(offset) / 2.0

			// Shift left image right and right image left
			leftImage = leftImage.transformed(by: CGAffineTransform(translationX: halfOffset, y: 0))
			rightImage = rightImage.transformed(by: CGAffineTransform(translationX: -halfOffset, y: 0))

			// Crop to overlapping area
			let cropRect = CGRect(
				x: abs(halfOffset),
				y: 0,
				width: halfWidth - abs(CGFloat(offset)),
				height: height
			)

			leftImage = leftImage.cropped(to: cropRect)
			rightImage = rightImage.cropped(to: cropRect)

			if verbose {
				print("  Output size after offset: \(Int(cropRect.width)) x \(Int(cropRect.height))")
			}
		}

		// Apply the anaglyph filter
		let outputImage = try AnaglyphFilter.apply(mode, left: leftImage, right: rightImage)

		// Determine output path
		let finalOutputPath = generateOutputPath(
			from: inputPath,
			outputDirectory: outputDirectory,
			useModeNaming: useModeNaming,
			mode: mode
		)

		if verbose {
			print("  Output: \(finalOutputPath)")
		}

		// Save the output
		try saveImage(outputImage, to: finalOutputPath, quality: quality)

		let processingTime = Date().timeIntervalSince(startTime)

		if verbose {
			print("  Processing time: \(String(format: "%.3f", processingTime)) seconds")
		}

		print("✓ Converted: \(URL(fileURLWithPath: inputPath).lastPathComponent) → \(URL(fileURLWithPath: finalOutputPath).lastPathComponent)")
	}

	private func generateOutputPath(
		from inputPath: String,
		outputDirectory: String?,
		useModeNaming: Bool,
		mode: AnaglyphFilter.Mode
	) -> String {
		let url = URL(fileURLWithPath: inputPath)
		let directory = outputDirectory.map { URL(fileURLWithPath: $0) } ?? url.deletingLastPathComponent()
		let nameWithoutExt = url.deletingPathExtension().lastPathComponent
		let ext = url.pathExtension

		let modeSuffix = useModeNaming ? mode.fileSuffix : "anaglyph"

		let outputName = "\(nameWithoutExt)-\(modeSuffix).\(ext)"
		return directory.appendingPathComponent(outputName).path
	}

	private func saveImage(_ image: CIImage, to path: String, quality: Float) throws {
		let url = URL(fileURLWithPath: path)
		let fileExtension = url.pathExtension.lowercased()

		// HEIC needs to be written through Core Image; NSBitmapImageRep can't encode it
		if fileExtension == "heic" {
			let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
			try ciContext.writeHEIFRepresentation(
				of: image,
				to: url,
				format: .RGBA8,
				colorSpace: colorSpace,
				options: [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): NSNumber(value: quality)]
			)
			printFileSize(at: url)
			return
		}

		// Create CGImage from CIImage
		guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
			throw AnaglyphError.failedToCreateCGImage
		}

		// Create NSBitmapImageRep
		let bitmapRep = NSBitmapImageRep(cgImage: cgImage)

		// Determine file type and properties
		let fileType: NSBitmapImageRep.FileType
		var properties: [NSBitmapImageRep.PropertyKey: Any] = [:]

		switch fileExtension {
		case "jpg", "jpeg":
			fileType = .jpeg
			properties[.compressionFactor] = NSNumber(value: quality)
		case "png":
			fileType = .png
		case "tiff", "tif":
			fileType = .tiff
			properties[.compressionMethod] = NSNumber(value: NSBitmapImageRep.TIFFCompression.lzw.rawValue)
		case "bmp":
			fileType = .bmp
		default:
			fileType = .jpeg
			properties[.compressionFactor] = NSNumber(value: quality)
		}

		// Generate data and write to file
		guard let data = bitmapRep.representation(using: fileType, properties: properties) else {
			throw AnaglyphError.failedToGenerateImageData
		}

		try data.write(to: url)
		printFileSize(at: url)
	}

	private func printFileSize(at url: URL) {
		guard verbose else { return }

		if let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int {
			let sizeInMB = Double(fileSize) / (1024 * 1024)
			print("  File size: \(String(format: "%.2f", sizeInMB)) MB")
		}
	}
}

// MARK: - Error Types

enum AnaglyphError: LocalizedError {
	case failedToLoadImage(String)
	case failedToCompileKernel(String)
	case failedToGenerateOutput
	case failedToCreateCGImage
	case failedToGenerateImageData

	var errorDescription: String? {
		switch self {
		case .failedToLoadImage(let path):
			return "Failed to load image: \(path)"
		case .failedToCompileKernel(let mode):
			return "Failed to compile Metal kernel for mode: \(mode)"
		case .failedToGenerateOutput:
			return "Failed to generate anaglyph output"
		case .failedToCreateCGImage:
			return "Failed to create CGImage for saving"
		case .failedToGenerateImageData:
			return "Failed to generate image data for saving"
		}
	}
}

// MARK: - Command Line Interface

func exitWithUsageError(_ message: String) -> Never {
	print("Error: \(message)")
	print("Use -h or --help for usage information.")
	exit(1)
}

struct CommandLineArgs {
	let inputPaths: [String]
	let outputDirectory: String?
	let mode: AnaglyphFilter.Mode
	let quality: Float
	let autoDetect: Bool
	let fastMode: Bool
	let name: Bool
	let manualOffset: Int?
	let verbose: Bool
	let help: Bool

	static func parse() -> CommandLineArgs {
		var inputPaths: [String] = []
		var outputDirectory: String?
		var mode: AnaglyphFilter.Mode = .simple
		var quality: Float = 0.9
		var autoDetect = false
		var fastMode = false
		var name = false
		var manualOffset: Int?
		var verbose = false
		var help = false

		let args = Array(CommandLine.arguments.dropFirst())
		var i = 0

		func requireValue(for arg: String) -> String {
			guard i + 1 < args.count else {
				exitWithUsageError("Missing value for \(arg)")
			}
			let value = args[i + 1]
			i += 2
			return value
		}

		while i < args.count {
			let arg = args[i]

			switch arg {
			case "-o", "--output":
				outputDirectory = requireValue(for: arg)

			case "-m", "--mode":
				let value = requireValue(for: arg)
				switch value.lowercased() {
				case "simple":
					mode = .simple
				case "optimized", "opt":
					mode = .optimized
				case "dubois", "color":
					mode = .dubois
				case "grayscale", "gray":
					mode = .grayscale
				default:
					exitWithUsageError("Unknown mode: \(value) (valid modes: simple, optimized, dubois, grayscale)")
				}

			case "-q", "--quality":
				let value = requireValue(for: arg)
				guard let q = Float(value), q >= 0, q <= 1 else {
					exitWithUsageError("Invalid quality '\(value)' — expected a number between 0 and 1")
				}
				quality = q

			case "-a", "--auto":
				autoDetect = true
				i += 1

			case "-f", "--fast":
				fastMode = true
				i += 1

			case "-n", "--name":
				name = true
				i += 1

			case "--offset":
				let value = requireValue(for: arg)
				guard let o = Int(value) else {
					exitWithUsageError("Invalid offset '\(value)' — expected an integer number of pixels")
				}
				manualOffset = o

			case "-v", "--verbose":
				verbose = true
				i += 1

			case "-h", "--help":
				help = true
				i += 1

			default:
				if arg.starts(with: "-") {
					// Bare negative numbers (e.g. -100) are shorthand for --offset
					guard let o = Int(arg) else {
						exitWithUsageError("Unknown option: \(arg)")
					}
					manualOffset = o
					i += 1
				} else if arg.starts(with: "+"), let o = Int(arg) {
					// Explicit positive offsets need a + prefix (a bare 100 would be a filename)
					manualOffset = o
					i += 1
				} else {
					inputPaths.append(arg)
					i += 1
				}
			}
		}

		return CommandLineArgs(
			inputPaths: inputPaths,
			outputDirectory: outputDirectory,
			mode: mode,
			quality: quality,
			autoDetect: autoDetect,
			fastMode: fastMode,
			name: name,
			manualOffset: manualOffset,
			verbose: verbose,
			help: help
		)
	}
}

func printHelp() {
	print("""
	Anaglyph Converter - Convert side-by-side stereoscopic images to red-cyan anaglyph
	With intelligent stereo disparity analysis for automatic convergence adjustment

	Usage: \(CommandLine.arguments[0]) [options] <image1> [image2] ...

	Options:
		-o, --output <dir>     Output directory (default: same as input)
		-m, --mode <mode>      Anaglyph mode (default: simple)
							   Modes: simple, optimized, dubois, grayscale
		-n, --name             Use mode-based naming (default: only append -anaglyph to file name)
							   mode-based naming appends -anaglyph-<mode> to file name
		-q, --quality <0-1>    JPEG compression quality (default: 0.9)
		-a, --auto             Auto-detect optimal offset using disparity analysis
		-f, --fast             Fast mode (fewer samples, less accurate but quicker)
		--offset <pixels>      Manual offset override (negative = closer convergence)
							   Bare signed numbers (e.g. -100 or +40) are shorthand for --offset
		-v, --verbose          Show detailed processing information
		-h, --help             Show this help message

	Modes:
		simple     - Basic red/cyan channel separation (fast, good depth)
		optimized  - Optimized matrices for better depth perception
		dubois     - Dubois method for better color preservation
		grayscale  - Grayscale anaglyph (reduces color rivalry)

	Auto-Detection:
		The -a flag analyzes the stereo pair to find the main subject depth
		and automatically sets the convergence to place it at screen depth.
		This works by:
		1. Finding matching features between left/right images
		2. Calculating disparity (depth) for each feature
		3. Identifying the main subject depth range
		4. Setting offset to bring main subject to zero disparity

	Manual Offset Guidelines:
		0         - No adjustment (default)
		-20 to -40  - Slight convergence for distant subjects
		-40 to -60  - Medium convergence for general scenes
		-80 to -100 - Strong convergence for close subjects
		-100+     - Maximum convergence for very close/macro subjects

	Examples:
		\(CommandLine.arguments[0]) photo.jpg                    # Basic conversion
		\(CommandLine.arguments[0]) -a photo.jpg                 # Auto-detect offset
		\(CommandLine.arguments[0]) -a -f photo.jpg              # Fast auto-detect
		\(CommandLine.arguments[0]) -a -v photo.jpg              # Auto with details
		\(CommandLine.arguments[0]) --offset -80 close.jpg       # Manual for close subject
		\(CommandLine.arguments[0]) -m dubois -a -f *.jpg        # Best quality, fast auto

	Supported formats: JPEG, PNG, TIFF, HEIC, BMP
	Output files are named: <input>-anaglyph.<ext>
	""")
}

// MARK: - Main

func main() {
	let args = CommandLineArgs.parse()

	if args.help || args.inputPaths.isEmpty {
		printHelp()
		exit(args.help ? 0 : 1)
	}

	// Create the output directory once, up front
	if let outputDir = args.outputDirectory {
		do {
			try FileManager.default.createDirectory(
				atPath: outputDir,
				withIntermediateDirectories: true
			)
		} catch {
			print("✗ Failed to create output directory \(outputDir): \(error.localizedDescription)")
			exit(1)
		}
	}

	let converter = AnaglyphConverter(verbose: args.verbose)
	var successCount = 0
	var failureCount = 0
	let startTime = Date()

	print("Anaglyph Converter (Metal Kernel)")
	if args.autoDetect {
		print("Mode: \(args.mode.rawValue) | Offset: Auto-detect")
	} else if let offset = args.manualOffset {
		print("Mode: \(args.mode.rawValue) | Offset: \(offset)")
	} else {
		print("Mode: \(args.mode.rawValue) | Offset: 0")
	}
	print("Processing \(args.inputPaths.count) image(s)...")

	for inputPath in args.inputPaths {
		do {
			try converter.processImage(
				at: inputPath,
				outputDirectory: args.outputDirectory,
				mode: args.mode,
				quality: args.quality,
				autoDetect: args.autoDetect,
				fastMode: args.fastMode,
				useModeNaming: args.name,
				manualOffset: args.manualOffset
			)
			successCount += 1
		} catch {
			print("✗ Error with \(inputPath): \(error.localizedDescription)")
			failureCount += 1
		}
	}

	let elapsed = Date().timeIntervalSince(startTime)
	print("\nCompleted in \(String(format: "%.2f", elapsed)) seconds")
	print("Results: \(successCount) succeeded, \(failureCount) failed")

	exit(failureCount > 0 ? 1 : 0)
}

main()
