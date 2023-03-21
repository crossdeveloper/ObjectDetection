//
//  ViewController.swift
//  ObjectDetection
//
//  Created by Nickolay Lamm on 3/20/23.
//

import UIKit
import Vision
import CoreMedia

class ViewController: UIViewController {

    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var boxesView: DrawingBoundingBoxView!
    
    /// YOLOv5 is generally faster and more accurate than YOLOv3FP16.
    /// YOLOv5 achieves state-of-the-art accuracy on several benchmark datasets while maintaining a fast inference speed.
    /// It also supports a wider range of input sizes and can detect smaller objects than YOLOv3FP16.
        
    private let objectDectectionModel = try? yolov5s(configuration: MLModelConfiguration())
    
    // MARK: - Vision Properties
    private var request: VNCoreMLRequest?
    private var visionModel: VNCoreMLModel?
    private var isInferencing = false
    
    // MARK: - AV Property
    private var videoCapture: VideoCapture!
    private let semaphore = DispatchSemaphore(value: 1)
    
    // MARK: - Person Object Property
    private var personObject: VNRecognizedObjectObservation?
    
    // MARK - Last Object Properties
    private var lastObjectLocation: CGRect?
    private var lastObjectTimestamp: Date?
    
    private let maf = MovingAverageFilter()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup the model
        setUpModel()
        
        // setup camera
        setUpCamera()
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        self.videoCapture.start()
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        self.videoCapture.stop()
    }
    
    // MARK: - Setup Core ML
    func setUpModel() {
        if let objectDectectionModel = objectDectectionModel, let visionModel = try? VNCoreMLModel(for: objectDectectionModel.model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .scaleFill
        } else {
            fatalError("fail to create vision model")
        }
    }

    // MARK: - SetUp Video
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.fps = 30
        videoCapture.setUp(sessionPreset: .hd1920x1080) { success in
            
            if success {
                // add preview view on the layer
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                
                // start video preview when setup is done
                self.videoCapture.start()
            }
        }
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        resizePreviewLayer()
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
}

// MARK: - VideoCaptureDelegate
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // the captured image from camera is contained on pixelBuffer
        if !self.isInferencing, let pixelBuffer = pixelBuffer {
            self.isInferencing = true
            
            // predict!
            self.predictUsingVision(pixelBuffer: pixelBuffer)
        }
    }
}

extension ViewController {
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        self.semaphore.wait()
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        try? handler.perform([request])
    }
    
    // MARK: - Post-processing


    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let predictions = request.results as? [VNRecognizedObjectObservation], let object = predictions.first(where: { $0.label == "person" }) {
            // Store the location and timestamp of the detected object
            let objectLocation = object.boundingBox
            let objectTimestamp = Date()
            
            var moving = true
            
            // Calculate the difference between the current and previous object locations
            if let lastLocation = lastObjectLocation,
               let lastTimestamp = lastObjectTimestamp {
                let (notMoving, centered) = checkMotion(current: objectLocation, previous: lastLocation, timestamp: objectTimestamp.timeIntervalSince1970 - lastTimestamp.timeIntervalSince1970)
                
                // Trigger the desired action if there has been no significant motion detected for a certain period of time
                moving = !(notMoving && centered)
            }
            
            personObject = object
            lastObjectLocation = objectLocation
            lastObjectTimestamp = objectTimestamp
            
            DispatchQueue.main.async {
                if let object = self.personObject {
                    self.boxesView.drawBoxs(with: object, moving: moving)
                    self.isInferencing = false
                }
            }
        } else {
            self.isInferencing = false
        }
        self.semaphore.signal()
    }

    func checkMotion(current: CGRect, previous: CGRect, timestamp: Double) -> (Bool, Bool) {
        // Calculate the difference between the current and previous object locations
        let deltaX = abs(current.midX - previous.midX)
        let deltaY = abs(current.midY - previous.midY)
        let deltaWidth = abs(current.width - previous.width)
        let deltaHeight = abs(current.height - previous.height)

        // Apply a weight to each component of the difference based on how much they contribute to the overall motion
        let weightedDeltaX = deltaX * 2.0
        let weightedDeltaY = deltaY * 1.0
        let weightedDeltaWidth = deltaWidth * 2.0
        let weightedDeltaHeight = deltaHeight * 1.0

        // Calculate the overall difference using a weighted sum
        let diff = (weightedDeltaX + weightedDeltaY + weightedDeltaWidth + weightedDeltaHeight) / timestamp
        maf.append(element: diff)
        maf.maxCount = Int(1.5 / timestamp)
        
        let centerMin = 0.35
        let centerMax = 0.65
        
        let notMoving = maf.averageValue < 0.1
        let centered = current.midX > centerMin && current.midX < centerMax
        
        return (notMoving, centered)
    }

}

class MovingAverageFilter {
    private var arr: [CGFloat] = []
    var maxCount = 10
    
    public func append(element: CGFloat) {
        arr.append(element)
        if arr.count > maxCount {
            arr.removeFirst()
        }
    }
    
    public var averageValue: CGFloat {
        guard !arr.isEmpty else { return 0 }
        let sum = arr.reduce(0) { $0 + $1 }
        return CGFloat(sum) / CGFloat(arr.count)
    }
}
