//
//  main.swift
//  anaglyphtool
//
//  Created by Scott Jann on 12/16/25.
//

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
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
		let suggestedOffset: Int         // Horizontal convergence shift, full-resolution pixels
		let verticalCorrection: Int      // Pixels to move the right eye up (+) or down (-) to align rows
		let nearDisparity: Float         // Disparity of the nearest reliable content (full-res px)
		let farDisparity: Float          // Disparity of the furthest reliable content (full-res px)
		let mainSubjectDisparity: Float  // Centre-weighted median disparity (full-res px)
		let confidence: Float
		let sampleCount: Int

		static let none = DisparityResult(
			suggestedOffset: 0, verticalCorrection: 0,
			nearDisparity: 0, farDisparity: 0, mainSubjectDisparity: 0,
			confidence: 0, sampleCount: 0
		)
	}

	// Grayscale pixels rendered once per image, safe to read from any thread
	private struct GrayscaleBuffer {
		let pixels: [UInt8]
		let width: Int
		let height: Int
	}

	private struct Match {
		let disparity: Float  // Analysis-scale px; positive = content sits further left in the right eye (nearer)
		let dy: Int           // Row offset of the right-eye match (analysis-scale px)
		let weight: Float     // Higher near the frame centre
	}

	// Matching tunables, all at analysis scale
	private static let blockRadius = 8               // 16 x 16 comparison block
	private static let minTexture: Float = 3.0       // Mean |horizontal gradient| a block needs to be matchable
	private static let maxMatchError: Float = 18.0   // Mean abs difference (after brightness compensation) to accept
	private static let uniquenessRatio: Float = 0.8  // Best match must beat the runner-up by this factor

	// Analyze a stereo pair and pick the convergence that places the nearest content at screen depth.
	// Everything else then sits behind the screen, which is the most comfortable arrangement for
	// red/cyan anaglyphs: crossed (in-front) parallax is where ghosting is most visible and fusion fails.
	static func analyzeDisparity(
		left: CIImage,
		right: CIImage,
		context: CIContext,
		verbose: Bool = false,
		fast: Bool = false
	) -> DisparityResult {

		let eyeWidth = left.extent.width
		guard eyeWidth > 0, left.extent.height > 0 else { return .none }

		// Matching runs on a reduced copy. Disparity scales linearly with image size, so a 1024-px
		// analysis image gives ~1 px of full-res precision per 5 px on a typical camera, and the
		// cost drops by the square of the scale factor. Fast mode halves the resolution again.
		let analysisWidth: CGFloat = fast ? 512 : 1024
		let scale = min(1, analysisWidth / eyeWidth)
		let gridStep = fast ? 10 : 12

		let startRender = Date()
		guard let leftBuffer = renderGrayscale(left, scale: scale, context: context),
			  let rightBuffer = renderGrayscale(right, scale: scale, context: context),
			  leftBuffer.width == rightBuffer.width,
			  leftBuffer.height == rightBuffer.height else {
			return .none
		}

		if verbose {
			print("    Analysis size: \(leftBuffer.width) x \(leftBuffer.height) (scale \(String(format: "%.3f", scale)))")
			print("    Image rendering: \(String(format: "%.2f", Date().timeIntervalSince(startRender)))s")
		}

		let (matches, candidateCount) = findMatches(
			left: leftBuffer,
			right: rightBuffer,
			gridStep: gridStep,
			verbose: verbose
		)

		guard matches.count >= 8 else {
			if verbose {
				print("    Too few reliable matches (\(matches.count)); leaving offset at 0")
			}
			return .none
		}

		let toFullRes = Float(1 / scale)
		let disparities = matches.map { $0.disparity * toFullRes }.sorted()
		// "Nearest" is the largest disparity still backed by a handful of samples, so a lone bad
		// match cannot set the convergence but a small foreground object still counts
		let support = max(5, disparities.count / 100)
		let near = disparities[disparities.count - support]
		let far = disparities[support - 1]
		let mainSubject = weightedMedian(matches) * toFullRes

		let dys = matches.map { Float($0.dy) }.sorted()
		let verticalCorrection = Int((percentile(dys, 0.5) * toFullRes).rounded())

		// Nearest content lands just behind the screen so objects touching the frame edge don't
		// appear to poke through the "window"
		let nearPad = Float(eyeWidth) * 0.003
		let suggestedOffset = Int((-(near + nearPad)).rounded())

		// Confidence: were enough blocks matchable, and is the near estimate supported by many samples?
		let acceptance = Float(matches.count) / Float(max(candidateCount, 1))
		let nearSupport = disparities.filter { $0 >= near - Float(eyeWidth) * 0.02 }.count
		let confidence = 0.5 * min(acceptance / 0.3, 1) + 0.5 * min(Float(nearSupport) / 40, 1)

		if verbose {
			let range = near - far
			print("\n  Disparity Analysis:")
			print("    Samples analyzed: \(matches.count) of \(candidateCount) blocks")
			print("    Near objects: \(Int(near)) pixels disparity")
			print("    Far objects: \(Int(far)) pixels disparity")
			print("    Main subject: \(Int(mainSubject)) pixels disparity")
			print("    Depth range: \(Int(range)) pixels (\(String(format: "%.1f%%", range / Float(eyeWidth) * 100)) of width)")
			if range / Float(eyeWidth) > 1.0 / 30.0 {
				print("    Note: depth range exceeds the 1/30 comfort guideline; view smaller or use a manual offset")
			}
			print("    Vertical misalignment: \(verticalCorrection) pixels")
			print("    Distribution (far -> near): \(histogram(disparities, from: far, to: near))")
			print("    Suggested offset: \(suggestedOffset) pixels (nearest content at screen depth)")
			print("    Confidence: \(String(format: "%.1f%%", confidence * 100))")
		}

		return DisparityResult(
			suggestedOffset: suggestedOffset,
			verticalCorrection: verticalCorrection,
			nearDisparity: near,
			farDisparity: far,
			mainSubjectDisparity: mainSubject,
			confidence: confidence,
			sampleCount: matches.count
		)
	}

	// Render a CIImage, downscaled, into a single-channel 8-bit grayscale buffer
	private static func renderGrayscale(_ image: CIImage, scale: CGFloat, context: CIContext) -> GrayscaleBuffer? {
		var scaled = image
		if scale < 1 {
			let filter = CIFilter.lanczosScaleTransform()
			filter.inputImage = image
			filter.scale = Float(scale)
			filter.aspectRatio = 1
			guard let output = filter.outputImage else { return nil }
			scaled = output
		}

		let extent = scaled.extent
		let width = Int(extent.width.rounded(.down))
		let height = Int(extent.height.rounded(.down))
		guard width > 0, height > 0 else { return nil }

		scaled = scaled.transformed(by: CGAffineTransform(translationX: -extent.origin.x, y: -extent.origin.y))

		var rgba = [UInt8](repeating: 0, count: width * height * 4)
		rgba.withUnsafeMutableBytes { buffer in
			context.render(
				scaled,
				toBitmap: buffer.baseAddress!,
				rowBytes: width * 4,
				bounds: CGRect(x: 0, y: 0, width: width, height: height),
				format: .RGBA8,
				colorSpace: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
			)
		}

		var pixels = [UInt8](repeating: 0, count: width * height)
		for i in 0..<(width * height) {
			let r = Int(rgba[i * 4])
			let g = Int(rgba[i * 4 + 1])
			let b = Int(rgba[i * 4 + 2])
			pixels[i] = UInt8((r * 299 + g * 587 + b * 114) / 1000)
		}

		return GrayscaleBuffer(pixels: pixels, width: width, height: height)
	}

	// Summed-area table so any block sum in the right eye costs four lookups
	private static func integralImage(_ buffer: GrayscaleBuffer) -> [Int32] {
		let stride = buffer.width + 1
		var table = [Int32](repeating: 0, count: stride * (buffer.height + 1))
		for y in 0..<buffer.height {
			var rowSum: Int32 = 0
			let rowBase = y * buffer.width
			for x in 0..<buffer.width {
				rowSum += Int32(buffer.pixels[rowBase + x])
				table[(y + 1) * stride + x + 1] = table[y * stride + x + 1] + rowSum
			}
		}
		return table
	}

	// Block-match a grid of sample points in parallel. Runs twice: a sparse pass with a wide
	// vertical search finds how far the two eyes are misaligned vertically, then the full grid
	// is matched with the vertical search centred on that answer. Matching with the wrong row
	// offset does not just fail, it locks onto neighbouring texture and reports a confident
	// but wrong disparity, so the vertical estimate has to come first.
	private static func findMatches(
		left: GrayscaleBuffer,
		right: GrayscaleBuffer,
		gridStep: Int,
		verbose: Bool
	) -> (matches: [Match], candidates: Int) {

		let width = left.width
		let height = left.height
		let coarseDy = 6  // Rows of misalignment the sparse pass can find (analysis scale)
		let margin = blockRadius + coarseDy + 2
		let searchRange = width / 5  // +/-20% of the eye width covers macro subjects and pre-converged pairs

		// Sample the central 80% of the frame to stay clear of edge artifacts
		var points: [(x: Int, y: Int)] = []
		for y in stride(from: height / 10, to: height * 9 / 10, by: gridStep) where y >= margin && y < height - margin {
			for x in stride(from: width / 10, to: width * 9 / 10, by: gridStep) where x >= margin && x < width - margin {
				points.append((x: x, y: y))
			}
		}

		if verbose {
			print("    Analyzing \(points.count) sample points, search range +/-\(searchRange) px...")
		}

		let startAnalysis = Date()
		let rightSums = integralImage(right)

		let coarsePoints = stride(from: 0, to: points.count, by: 6).map { points[$0] }
		let coarse = matchGrid(
			points: coarsePoints, left: left, right: right, rightSums: rightSums,
			searchRange: searchRange, dyRange: -coarseDy...coarseDy
		)
		let rowOffset = coarse.count >= 8 ? Int(percentile(coarse.map { Float($0.dy) }.sorted(), 0.5)) : 0

		let matches = matchGrid(
			points: points, left: left, right: right, rightSums: rightSums,
			searchRange: searchRange, dyRange: (rowOffset - 1)...(rowOffset + 1)
		)

		if verbose {
			print("    Block matching: \(String(format: "%.2f", Date().timeIntervalSince(startAnalysis)))s (row offset \(rowOffset) from \(coarse.count) coarse matches)")
		}

		return (matches, points.count)
	}

	private static func matchGrid(
		points: [(x: Int, y: Int)],
		left: GrayscaleBuffer,
		right: GrayscaleBuffer,
		rightSums: [Int32],
		searchRange: Int,
		dyRange: ClosedRange<Int>
	) -> [Match] {

		let width = left.width
		let height = left.height
		let centerX = Float(width) / 2
		let centerY = Float(height) / 2

		// Each iteration writes only to its own slot. The matcher takes raw pointers because
		// buffer-pointer subscripts are bounds-checked in Debug builds, which made auto mode
		// take minutes on large images.
		var results = [Match?](repeating: nil, count: points.count)
		left.pixels.withUnsafeBufferPointer { leftPixels in
			right.pixels.withUnsafeBufferPointer { rightPixels in
				rightSums.withUnsafeBufferPointer { sums in
					results.withUnsafeMutableBufferPointer { output in
						DispatchQueue.concurrentPerform(iterations: points.count) { i in
							let point = points[i]
							guard let (disparity, dy) = matchBlock(
								x: point.x, y: point.y,
								left: leftPixels.baseAddress!, right: rightPixels.baseAddress!, rightSums: sums.baseAddress!,
								width: width, height: height,
								searchRange: searchRange, dyRange: dyRange
							) else { return }

							// Gaussian falloff from the frame centre, sigma = 35% of the half-size
							let nx = (Float(point.x) - centerX) / centerX
							let ny = (Float(point.y) - centerY) / centerY
							let weight = expf(-(nx * nx + ny * ny) / (2 * 0.35 * 0.35))
							output[i] = Match(disparity: disparity, dy: dy, weight: weight)
						}
					}
				}
			}
		}

		return results.compactMap { $0 }
	}

	// Find the disparity of one block, or nil if the block is flat, ambiguous, or has no good match
	private static func matchBlock(
		x: Int, y: Int,
		left: UnsafePointer<UInt8>,
		right: UnsafePointer<UInt8>,
		rightSums: UnsafePointer<Int32>,
		width: Int, height: Int,
		searchRange: Int, dyRange: ClosedRange<Int>
	) -> (disparity: Float, dy: Int)? {

		let side = 2 * blockRadius
		let n = side * side
		let x0 = x - blockRadius
		let y0 = y - blockRadius

		// Flat blocks (sky, backdrop, bokeh) match everywhere and would otherwise flood the
		// statistics with bogus zero disparities. Horizontal gradient is what a horizontal
		// search can lock onto, so that is what we require.
		var leftSum = 0
		var gradient = 0
		for row in 0..<side {
			let base = (y0 + row) * width + x0
			var previous = Int(left[base])
			leftSum += previous
			for col in 1..<side {
				let value = Int(left[base + col])
				leftSum += value
				gradient += abs(value - previous)
				previous = value
			}
		}
		guard Float(gradient) / Float(side * (side - 1)) >= minTexture else { return nil }

		// Candidate disparities d place the right-eye block at x0 - d; keep it inside the image
		let dMin = max(-searchRange, x0 + side - width)
		let dMax = min(searchRange, x0)
		guard dMax - dMin >= 8 else { return nil }
		let count = dMax - dMin + 1

		var cost = [Float](repeating: .infinity, count: count)
		var costDy = [Int](repeating: 0, count: count)
		let sumStride = width + 1

		// Zero-mean SAD: the two halves of a single-sensor stereo shot often differ in
		// brightness, so compare each block with its own mean removed. Summation stops
		// early once it passes `limit`, since a candidate that expensive can neither win
		// nor matter for the uniqueness test.
		func blockDifference(rx0: Int, ry0: Int, limit: Int) -> Int {
			let rightSum = Int(rightSums[(ry0 + side) * sumStride + rx0 + side])
				- Int(rightSums[ry0 * sumStride + rx0 + side])
				- Int(rightSums[(ry0 + side) * sumStride + rx0])
				+ Int(rightSums[ry0 * sumStride + rx0])
			let bias = (leftSum - rightSum) / n

			var sad = 0
			for row in 0..<side {
				let leftBase = (y0 + row) * width + x0
				let rightBase = (ry0 + row) * width + rx0
				for col in 0..<side {
					sad += abs(Int(left[leftBase + col]) - Int(right[rightBase + col]) - bias)
				}
				if sad > limit { break }
			}
			return sad
		}

		let acceptLimit = Int(maxMatchError * Float(n))
		var bestTotal = acceptLimit
		for dy in dyRange {
			let ry0 = y0 + dy
			guard ry0 >= 0, ry0 + side <= height else { continue }

			for d in dMin...dMax {
				let limit = Int(Float(bestTotal) / uniquenessRatio) + 1
				let sad = blockDifference(rx0: x0 - d, ry0: ry0, limit: limit)
				guard sad <= limit else { continue }

				let score = Float(sad) / Float(n)
				let index = d - dMin
				if score < cost[index] {
					cost[index] = score
					costDy[index] = dy
				}
				if sad < bestTotal {
					bestTotal = sad
				}
			}
		}

		var bestIndex = 0
		var best = Float.infinity
		for i in 0..<count where cost[i] < best {
			best = cost[i]
			bestIndex = i
		}
		guard best <= maxMatchError else { return nil }

		// Uniqueness: repetitive textures (brick, foliage, railings) produce several near-equal
		// minima, and picking one of them is a coin toss
		var runnerUp = Float.infinity
		for i in 0..<count where abs(i - bestIndex) >= 4 && cost[i] < runnerUp {
			runnerUp = cost[i]
		}
		guard best <= runnerUp * uniquenessRatio else { return nil }

		// Parabolic sub-pixel refinement around the minimum, on exact neighbour costs
		var disparity = Float(bestIndex + dMin)
		if bestIndex > 0 && bestIndex < count - 1 {
			let ry0 = y0 + costDy[bestIndex]
			let c0 = Float(blockDifference(rx0: x0 - (bestIndex + dMin - 1), ry0: ry0, limit: .max)) / Float(n)
			let c1 = cost[bestIndex]
			let c2 = Float(blockDifference(rx0: x0 - (bestIndex + dMin + 1), ry0: ry0, limit: .max)) / Float(n)
			let denominator = c0 - 2 * c1 + c2
			if denominator > 0 {
				disparity += 0.5 * (c0 - c2) / denominator
			}
		}

		return (disparity, costDy[bestIndex])
	}

	// Twelve-bin text histogram, e.g. "▂▃▇▅▁▁▁▁▁▁▁▂"
	private static func histogram(_ sorted: [Float], from low: Float, to high: Float) -> String {
		let bins = 12
		guard high > low, !sorted.isEmpty else { return "flat" }
		var counts = [Int](repeating: 0, count: bins)
		for value in sorted {
			let bin = Int((value - low) / (high - low) * Float(bins))
			counts[min(max(bin, 0), bins - 1)] += 1
		}
		let bars: [Character] = Array(" ▁▂▃▄▅▆▇█")
		let peak = Float(counts.max() ?? 1)
		return String(counts.map { bars[min(Int(Float($0) / peak * 8), 8)] })
	}

	private static func percentile(_ sorted: [Float], _ p: Float) -> Float {
		guard !sorted.isEmpty else { return 0 }
		let index = Int((Float(sorted.count - 1) * p).rounded())
		return sorted[min(max(index, 0), sorted.count - 1)]
	}

	// Median disparity with samples near the frame centre counting for more
	private static func weightedMedian(_ matches: [Match]) -> Float {
		let sorted = matches.sorted { $0.disparity < $1.disparity }
		let total = sorted.reduce(Float(0)) { $0 + $1.weight }
		var accumulated: Float = 0
		for match in sorted {
			accumulated += match.weight
			if accumulated >= total / 2 {
				return match.disparity
			}
		}
		return sorted.last?.disparity ?? 0
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
		// Whole-pixel eye size so the split and crops never land on a half pixel
		let eyeWidth = Int(width / 2)
		let eyeHeight = Int(height)
		let halfWidth = CGFloat(eyeWidth)

		if verbose {
			print("  Input dimensions: \(Int(width)) x \(Int(height))")
			print("  Each eye: \(eyeWidth) x \(eyeHeight)")
		}

		// Extract left and right images
		let leftRect = CGRect(x: 0, y: 0, width: halfWidth, height: CGFloat(eyeHeight))
		let rightRect = CGRect(x: halfWidth, y: 0, width: halfWidth, height: CGFloat(eyeHeight))

		var leftImage = ciImage.cropped(to: leftRect)
		var rightImage = ciImage.cropped(to: rightRect)

		// Move right image to same position as left
		rightImage = rightImage.transformed(by: CGAffineTransform(translationX: -halfWidth, y: 0))

		// Determine the horizontal offset and any vertical correction
		let offset: Int
		var verticalCorrection = 0
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
			verticalCorrection = disparityResult.verticalCorrection

			let vertical = verticalCorrection != 0 ? ", vertical correction: \(verticalCorrection) pixels" : ""
			print("  ✓ Auto-detected offset: \(offset) pixels\(vertical) (confidence: \(String(format: "%.0f%%", disparityResult.confidence * 100)))")
		} else {
			offset = 0
			if verbose {
				print("  No offset applied")
			}
		}

		if offset != 0 || verticalCorrection != 0 {
			// Split the offset between both eyes for centred convergence, in whole pixels: a
			// half-pixel translation would resample (soften) both eyes, and sharp edges are what
			// the viewer fuses on
			let shiftLeft = offset / 2
			let shiftRight = -(offset - shiftLeft)

			leftImage = leftImage.transformed(by: CGAffineTransform(translationX: CGFloat(shiftLeft), y: 0))
			rightImage = rightImage.transformed(by: CGAffineTransform(
				translationX: CGFloat(shiftRight),
				y: CGFloat(verticalCorrection)
			))

			// Crop to the area both eyes still cover
			let x0 = max(shiftLeft, shiftRight)
			let x1 = min(shiftLeft, shiftRight) + eyeWidth
			let y0 = max(0, verticalCorrection)
			let y1 = min(0, verticalCorrection) + eyeHeight
			guard x1 > x0, y1 > y0 else {
				throw AnaglyphError.offsetTooLarge(offset, eyeWidth)
			}

			let cropRect = CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
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
		let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
		let qualityOption = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)

		// Core Image encodes straight from its own render; bouncing through CGImage and
		// NSBitmapImageRep copies the full-size output twice more
		switch fileExtension {
		case "heic":
			try ciContext.writeHEIFRepresentation(
				of: image, to: url, format: .RGBA8, colorSpace: colorSpace,
				options: [qualityOption: NSNumber(value: quality)]
			)

		case "png":
			try ciContext.writePNGRepresentation(of: image, to: url, format: .RGBA8, colorSpace: colorSpace)

		case "tiff", "tif", "bmp":
			// Core Image has no BMP writer and writes uncompressed TIFF, so these still go through AppKit
			guard let cgImage = ciContext.createCGImage(image, from: image.extent) else {
				throw AnaglyphError.failedToCreateCGImage
			}
			let bitmapRep = NSBitmapImageRep(cgImage: cgImage)
			let isTIFF = fileExtension != "bmp"
			let properties: [NSBitmapImageRep.PropertyKey: Any] = isTIFF
				? [.compressionMethod: NSNumber(value: NSBitmapImageRep.TIFFCompression.lzw.rawValue)]
				: [:]
			guard let data = bitmapRep.representation(using: isTIFF ? .tiff : .bmp, properties: properties) else {
				throw AnaglyphError.failedToGenerateImageData
			}
			try data.write(to: url)

		default:
			try ciContext.writeJPEGRepresentation(
				of: image, to: url, colorSpace: colorSpace,
				options: [qualityOption: NSNumber(value: quality)]
			)
		}

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
	case offsetTooLarge(Int, Int)

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
		case .offsetTooLarge(let offset, let eyeWidth):
			return "Offset of \(offset) pixels leaves no overlap between eyes \(eyeWidth) pixels wide"
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
		-a, --auto             Auto-detect offset and vertical alignment from the stereo pair
		-f, --fast             Fast mode (lower analysis resolution, coarser result)
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
		The -a flag block-matches the two views (on a reduced copy, so it is
		fast even for very large images) and sets the convergence so the
		nearest content sits at screen depth. Everything else then appears
		behind the screen, where red/cyan ghosting is least visible. Any
		vertical misalignment between the eyes is measured and corrected too.
		Use -v to see near/far disparity, depth range and a histogram.

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
