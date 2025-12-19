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
import Accelerate

// MARK: - Custom Anaglyph Filter using Metal Kernel

class AnaglyphFilter: CIFilter {
	enum Mode {
		case simple           // Simple R/GB separation
		case optimized        // Optimized with better color
		case dubois          // Dubois method for color preservation
		case grayscale       // Grayscale anaglyph
		
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
		
		var kernel: CIKernel {
			return try! CIColorKernel.kernels(withMetalString: kernelString)[0]
		}
	}
	
	@objc dynamic var inputLeftImage: CIImage?
	@objc dynamic var inputRightImage: CIImage?
	var mode: Mode = .simple
	
	override var outputImage: CIImage? {
		guard let leftImage = inputLeftImage,
			  let rightImage = inputRightImage else {
			return nil
		}
		
		return mode.kernel.apply(
			extent: leftImage.extent,
			roiCallback: { _, rect in rect },
			arguments: [leftImage, rightImage]
		)
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
	
	// Analyze stereo pair to find optimal convergence
	static func analyzeDisparity(
		left: CIImage,
		right: CIImage,
		context: CIContext,
		verbose: Bool = false,
		fast: Bool = false
	) -> DisparityResult {
		
		// Convert to grayscale for analysis
		let leftGray = left.applyingFilter("CIPhotoEffectMono")
		let rightGray = right.applyingFilter("CIPhotoEffectMono")
		
		// Adjust grid size for fast mode
		let gridSize = fast ? 40 : 20  // Larger grid = fewer samples = faster
		
		// Find feature points and their disparities
		let disparities = findDisparitiesFast(
			left: leftGray,
			right: rightGray,
			context: context,
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
		let mainSubjectDisparity = findMainSubjectDisparity(
			disparities: disparities,
			imageSize: left.extent.size
		)
		
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
	
	// Optimized disparity finding with configurable grid size
	private static func findDisparitiesFast(
		left: CIImage,
		right: CIImage,
		context: CIContext,
		gridSize: Int,
		verbose: Bool
	) -> [Float] {
		
		// Sample points in a grid pattern
		let blockSize = 16  // Size of matching block
		let searchRange = Int(left.extent.width * 0.1)  // Reduced to 10% for speed
		
		let width = Int(left.extent.width)
		let height = Int(left.extent.height)
		
		// Focus on center 80% to avoid edge artifacts
		let startX = width / 10
		let endX = width * 9 / 10
		let startY = height / 10
		let endY = height * 9 / 10
		
		// Sample points
		var samplePoints: [(x: Int, y: Int)] = []
		for y in stride(from: startY, to: endY, by: gridSize) {
			for x in stride(from: startX, to: endX, by: gridSize) {
				samplePoints.append((x: x, y: y))
			}
		}
		
		if verbose {
			print("    Analyzing \(samplePoints.count) sample points...")
		}
		
		// Use a concurrent queue with a barrier for thread-safe array access
		var disparities: [Float] = []
		let disparityQueue = DispatchQueue(label: "disparity.queue", attributes: .concurrent)
		let dispatchGroup = DispatchGroup()
		let queue = DispatchQueue.global(qos: .userInitiated)
		let semaphore = DispatchSemaphore(value: ProcessInfo.processInfo.activeProcessorCount * 2)

		// Pre-render images to CGImages for faster pixel access
		let startRender = Date()
		let leftCG = context.createCGImage(left, from: left.extent)
		let rightCG = context.createCGImage(right, from: right.extent)
		
		if verbose {
			print("    Image rendering: \(String(format: "%.2f", Date().timeIntervalSince(startRender)))s")
		}
		
		let startAnalysis = Date()
		
		// Process points in parallel
		for point in samplePoints {
			dispatchGroup.enter()
			semaphore.wait()
			
			queue.async {
				defer {
					semaphore.signal()
					dispatchGroup.leave()
				}
				
				if let disparity = self.findPointDisparityFast(
					point: point,
					leftCG: leftCG,
					rightCG: rightCG,
					left: left,
					right: right,
					context: context,
					blockSize: blockSize,
					searchRange: searchRange
				) {
					// Thread-safe append using barrier
					disparityQueue.async(flags: .barrier) {
						disparities.append(disparity)
					}
				}
			}
		}
		
		// Wait for all computations to complete
		dispatchGroup.wait()
		
		if verbose {
			print("    Block matching: \(String(format: "%.2f", Date().timeIntervalSince(startAnalysis)))s")
		}
		
		// Return the collected disparities
		return disparities
	}
	
	// Find disparities using block matching (parallelized)
	private static func findDisparities(
		left: CIImage,
		right: CIImage,
		context: CIContext,
		verbose: Bool
	) -> [Float] {
		
		// Sample points in a grid pattern
		let gridSize = 20  // Sample every N pixels
		let blockSize = 16  // Size of matching block
		let searchRange = Int(left.extent.width * 0.15)  // Max 15% of width
		
		let width = Int(left.extent.width)
		let height = Int(left.extent.height)
		
		// Focus on center 80% to avoid edge artifacts
		let startX = width / 10
		let endX = width * 9 / 10
		let startY = height / 10
		let endY = height * 9 / 10
		
		// Sample points
		var samplePoints: [(x: Int, y: Int)] = []
		for y in stride(from: startY, to: endY, by: gridSize) {
			for x in stride(from: startX, to: endX, by: gridSize) {
				samplePoints.append((x: x, y: y))
			}
		}
		
		if verbose {
			print("    Sampling \(samplePoints.count) points (parallelized)...")
		}
		
		// Use a concurrent queue with a barrier for thread-safe array access
		var disparities: [Float] = []
		let disparityQueue = DispatchQueue(label: "disparity.queue", attributes: .concurrent)
		let dispatchGroup = DispatchGroup()
		let queue = DispatchQueue.global(qos: .userInitiated)
		let semaphore = DispatchSemaphore(value: ProcessInfo.processInfo.activeProcessorCount * 2)

		// Pre-render images to CGImages for faster pixel access
		let leftCG = context.createCGImage(left, from: left.extent)
		let rightCG = context.createCGImage(right, from: right.extent)
		
		// Process points in parallel
		for point in samplePoints {
			dispatchGroup.enter()
			semaphore.wait()
			
			queue.async {
				defer {
					semaphore.signal()
					dispatchGroup.leave()
				}
				
				if let disparity = self.findPointDisparityFast(
					point: point,
					leftCG: leftCG,
					rightCG: rightCG,
					left: left,
					right: right,
					context: context,
					blockSize: blockSize,
					searchRange: searchRange
				) {
					// Thread-safe append using barrier
					disparityQueue.async(flags: .barrier) {
						disparities.append(disparity)
					}
				}
			}
		}
		
		// Wait for all computations to complete
		dispatchGroup.wait()
		
		// Return the collected disparities
		return disparities
	}
	
	// Optimized disparity calculation using pre-rendered CGImages
	private static func findPointDisparityFast(
		point: (x: Int, y: Int),
		leftCG: CGImage?,
		rightCG: CGImage?,
		left: CIImage,
		right: CIImage,
		context: CIContext,
		blockSize: Int,
		searchRange: Int
	) -> Float? {
		
		// Use CGImages if available for faster pixel access
		if let leftCG = leftCG, let rightCG = rightCG {
			return findPointDisparityFromCGImage(
				point: point,
				leftCG: leftCG,
				rightCG: rightCG,
				blockSize: blockSize,
				searchRange: searchRange
			)
		}
		
		// Fallback to original CIImage method
		return findPointDisparity(
			point: point,
			left: left,
			right: right,
			context: context,
			blockSize: blockSize,
			searchRange: searchRange
		)
	}
	
	// Fast disparity calculation directly from CGImage pixels
	private static func findPointDisparityFromCGImage(
		point: (x: Int, y: Int),
		leftCG: CGImage,
		rightCG: CGImage,
		blockSize: Int,
		searchRange: Int
	) -> Float? {
		
		// Check bounds
		let halfBlock = blockSize / 2
		guard point.x >= halfBlock + searchRange,
			  point.x < leftCG.width - halfBlock,
			  point.y >= halfBlock,
			  point.y < leftCG.height - halfBlock else {
			return nil
		}
		
		// Get pixel data - create local copies to avoid retention issues
		guard let leftProvider = leftCG.dataProvider,
			  let rightProvider = rightCG.dataProvider,
			  let leftCFData = leftProvider.data,
			  let rightCFData = rightProvider.data else {
			return nil
		}
		
		// Convert to Data for safe access
		let leftData = leftCFData as Data
		let rightData = rightCFData as Data

		let leftBytesPerRow = leftCG.bytesPerRow
		let rightBytesPerRow = rightCG.bytesPerRow
		let bytesPerPixel = leftCG.bitsPerPixel / 8
		
		var bestDisparity: Float = 0
		var bestScore: Float = Float.infinity
		
		// Search for best match
		for xOffset in stride(from: 0, through: searchRange, by: 2) {
			var sum: Float = 0
			var count: Float = 0
			
			// Compare blocks
			for dy in -halfBlock..<halfBlock {
				for dx in -halfBlock..<halfBlock {
					let leftX = point.x + dx
					let leftY = point.y + dy
					let rightX = point.x + dx - xOffset
					let rightY = point.y + dy
					
					// Check bounds
					guard rightX >= 0 else { continue }

					// Calculate pixel indices
					let leftIndex = leftY * leftBytesPerRow + leftX * bytesPerPixel
					let rightIndex = rightY * rightBytesPerRow + rightX * bytesPerPixel
					
					// Ensure we're within bounds
					if leftIndex >= 0 && leftIndex + 2 < leftData.count &&
						rightIndex >= 0 && rightIndex + 2 < rightData.count {
						// Calculate difference in grayscale
						let leftGray = Float(leftData[leftIndex]) * 0.299 +
						Float(leftData[leftIndex + 1]) * 0.587 +
						Float(leftData[leftIndex + 2]) * 0.114
						let rightGray = Float(rightData[rightIndex]) * 0.299 +
						Float(rightData[rightIndex + 1]) * 0.587 +
						Float(rightData[rightIndex + 2]) * 0.114
						
						sum += abs(leftGray - rightGray)
						count += 1
					}
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
		
		// Only return if we have a confident match (low difference)
		return bestScore < 50 ? bestDisparity : nil
	}
	
	// Find disparity for a single point using block matching
	private static func findPointDisparity(
		point: (x: Int, y: Int),
		left: CIImage,
		right: CIImage,
		context: CIContext,
		blockSize: Int,
		searchRange: Int
	) -> Float? {
		
		// Extract block from left image
		let blockRect = CGRect(
			x: point.x - blockSize/2,
			y: point.y - blockSize/2,
			width: blockSize,
			height: blockSize
		)
		
		guard blockRect.minX >= 0,
			  blockRect.maxX <= left.extent.width else {
			return nil
		}
		
		let leftBlock = left.cropped(to: blockRect)
		
		// Search for best match in right image
		var bestDisparity: Float = 0
		var bestScore: Float = -Float.infinity
		
		// Search along epipolar line (same y-coordinate)
		for xOffset in stride(from: 0, through: searchRange, by: 1) {
			let rightRect = CGRect(
				x: point.x - xOffset - blockSize/2,
				y: point.y - blockSize/2,
				width: blockSize,
				height: blockSize
			)
			
			guard rightRect.minX >= 0 else { continue }
			
			let rightBlock = right.cropped(to: rightRect)
			
			// Calculate similarity using normalized cross-correlation
			let score = calculateBlockSimilarity(
				leftBlock: leftBlock,
				rightBlock: rightBlock,
				context: context
			)
			
			if score > bestScore {
				bestScore = score
				bestDisparity = Float(xOffset)
			}
		}
		
		// Only return if we have a confident match
		return bestScore > 0.7 ? bestDisparity : nil
	}
	
	// Calculate similarity between two blocks
	private static func calculateBlockSimilarity(
		leftBlock: CIImage,
		rightBlock: CIImage,
		context: CIContext
	) -> Float {
		
		// Use difference blend mode for quick comparison
		let difference = leftBlock.applyingFilter("CIDifferenceBlendMode", parameters: [
			kCIInputBackgroundImageKey: rightBlock
		])
		
		// Calculate average pixel difference
		guard let cgImage = context.createCGImage(difference, from: leftBlock.extent) else {
			return 0
		}
		
		guard let data = cgImage.dataProvider?.data as Data? else {
			return 0
		}
		
		var sum: Float = 0
		let pixelCount = data.count / 4
		
		for i in stride(from: 0, to: data.count, by: 4) {
			if i + 2 < data.count {
				let r = Float(data[i]) / 255.0
				let g = Float(data[i + 1]) / 255.0
				let b = Float(data[i + 2]) / 255.0
				let luminance = r * 0.299 + g * 0.587 + b * 0.114
				sum += 1.0 - luminance  // Invert so lower difference = higher score
			}
		}
		
		return pixelCount > 0 ? sum / Float(pixelCount) : 0
	}
	
	// Find the main subject disparity using heuristics
	private static func findMainSubjectDisparity(
		disparities: [Float],
		imageSize: CGSize
	) -> Float {
		
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
		outputPath: String? = nil,
		mode: AnaglyphFilter.Mode = .simple,
		quality: Float = 0.9,
		autoDetect: Bool = false,
		fastMode: Bool = false,
		useModeNaming: Bool = false,
		manualOffset: Int? = nil
	) throws {
		if verbose {
			print("\nProcessing: \(inputPath)")
			print("  Mode: \(mode)")
		}
		
		let startTime = Date()
		
		// Load the image
		guard let inputImage = NSImage(contentsOfFile: inputPath) else {
			throw AnaglyphError.failedToLoadImage(inputPath)
		}
		
		// Convert to CIImage
		guard let tiffData = inputImage.tiffRepresentation,
			  let bitmap = NSBitmapImageRep(data: tiffData),
			  let cgImage = bitmap.cgImage else {
			throw AnaglyphError.failedToCreateCIImage
		}
		
		let ciImage = CIImage(cgImage: cgImage)
		
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
			
			if verbose || autoDetect {
				print("  ✓ Auto-detected offset: \(offset) pixels (confidence: \(String(format: "%.0f%%", disparityResult.confidence * 100)))")
			}
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
		let anaglyphFilter = AnaglyphFilter()
		anaglyphFilter.inputLeftImage = leftImage
		anaglyphFilter.inputRightImage = rightImage
		anaglyphFilter.mode = mode
		
		guard let outputImage = anaglyphFilter.outputImage else {
			throw AnaglyphError.failedToGenerateOutput
		}
		
		// Determine output path
		let finalOutputPath = outputPath ?? generateOutputPath(from: inputPath, useModeNaming: useModeNaming, mode: mode)
		
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
	
	private func generateOutputPath(from inputPath: String, useModeNaming: Bool, mode: AnaglyphFilter.Mode) -> String {
		let url = URL(fileURLWithPath: inputPath)
		let directory = url.deletingLastPathComponent()
		let nameWithoutExt = url.deletingPathExtension().lastPathComponent
		let ext = url.pathExtension
		
		let modeSuffix: String
		if useModeNaming {
			switch mode {
			case .simple:
				modeSuffix = "anaglyph"
			case .optimized:
				modeSuffix = "anaglyph-opt"
			case .dubois:
				modeSuffix = "anaglyph-dubois"
			case .grayscale:
				modeSuffix = "anaglyph-gray"
			}
		} else {
			modeSuffix = "anaglyph"
		}
		
		let outputName = "\(nameWithoutExt)-\(modeSuffix).\(ext)"
		return directory.appendingPathComponent(outputName).path
	}
	
	private func saveImage(_ image: CIImage, to path: String, quality: Float) throws {
		let url = URL(fileURLWithPath: path)
		let fileExtension = url.pathExtension.lowercased()
		
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
		
		if verbose {
			let fileSize = data.count
			let sizeInMB = Double(fileSize) / (1024 * 1024)
			print("  File size: \(String(format: "%.2f", sizeInMB)) MB")
		}
	}
}

// MARK: - Error Types

enum AnaglyphError: LocalizedError {
	case failedToLoadImage(String)
	case failedToCreateCIImage
	case failedToGenerateOutput
	case failedToCreateCGImage
	case failedToGenerateImageData
	
	var errorDescription: String? {
		switch self {
		case .failedToLoadImage(let path):
			return "Failed to load image: \(path)"
		case .failedToCreateCIImage:
			return "Failed to create CIImage from input"
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
		
		while i < args.count {
			let arg = args[i]
			
			switch arg {
			case "-o", "--output":
				if i + 1 < args.count {
					outputDirectory = args[i + 1]
					i += 2
				} else {
					i += 1
				}
				
			case "-m", "--mode":
				if i + 1 < args.count {
					switch args[i + 1].lowercased() {
					case "simple":
						mode = .simple
					case "optimized", "opt":
						mode = .optimized
					case "dubois", "color":
						mode = .dubois
					case "grayscale", "gray":
						mode = .grayscale
					default:
						print("Unknown mode: \(args[i + 1])")
						print("Valid modes: simple, optimized, dubois, grayscale")
					}
					i += 2
				} else {
					i += 1
				}
				
			case "-q", "--quality":
				if i + 1 < args.count, let q = Float(args[i + 1]), q >= 0, q <= 1 {
					quality = q
					i += 2
				} else {
					i += 1
				}
				
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
				if i + 1 < args.count, let o = Int(args[i + 1]) {
					manualOffset = o
					i += 2
				} else {
					i += 1
				}
				
			case "-v", "--verbose":
				verbose = true
				i += 1
				
			case "-h", "--help":
				help = true
				i += 1
				
			default:
				if !arg.starts(with: "-") {
					inputPaths.append(arg)
				}
				i += 1
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
	
	let converter = AnaglyphConverter(verbose: args.verbose)
	var successCount = 0
	var failureCount = 0
	let startTime = Date()
	
	print("Anaglyph Converter (Metal Kernel)")
	if args.autoDetect {
		print("Mode: \(args.mode) | Offset: Auto-detect")
	} else if let offset = args.manualOffset {
		print("Mode: \(args.mode) | Offset: \(offset)")
	} else {
		print("Mode: \(args.mode) | Offset: 0")
	}
	print("Processing \(args.inputPaths.count) image(s)...")
	
	for inputPath in args.inputPaths {
		do {
			let outputPath: String?
			if let outputDir = args.outputDirectory {
				// Create output directory if needed
				try? FileManager.default.createDirectory(
					atPath: outputDir,
					withIntermediateDirectories: true
				)
				
				let inputURL = URL(fileURLWithPath: inputPath)
				let nameWithoutExt = inputURL.deletingPathExtension().lastPathComponent
				let ext = inputURL.pathExtension
				
				let modeSuffix: String
				if args.name {
					switch args.mode {
					case .simple:
						modeSuffix = "anaglyph"
					case .optimized:
						modeSuffix = "anaglyph-opt"
					case .dubois:
						modeSuffix = "anaglyph-dubois"
					case .grayscale:
						modeSuffix = "anaglyph-gray"
					}
				} else {
					modeSuffix = "anaglyph"
				}
				
				outputPath = "\(outputDir)/\(nameWithoutExt)-\(modeSuffix).\(ext)"
			} else {
				outputPath = nil
			}
			
			try converter.processImage(
				at: inputPath,
				outputPath: outputPath,
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
