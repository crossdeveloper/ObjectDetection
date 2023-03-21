//
//  DrawingBoundingBoxView.swift
//  ObjectDetection
//
//  Created by Nickolay Lamm on 3/20/23.
//

import UIKit
import Vision

class DrawingBoundingBoxView: UIView {
    func drawBoxs(with prediction: VNRecognizedObjectObservation, moving: Bool) {
        subviews.forEach({ $0.removeFromSuperview() })
        
        createLabelAndBox(prediction: prediction, moving: moving)
        self.setNeedsDisplay()
    }
    
    func createLabelAndBox(prediction: VNRecognizedObjectObservation, moving: Bool) {
        let color: UIColor = moving ? .red : .green
        
        let scale = CGAffineTransform.identity.scaledBy(x: bounds.width, y: bounds.height)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -1)
        let bgRect = prediction.boundingBox.applying(transform).applying(scale)
        
        let bgView = UIView(frame: bgRect)
        bgView.layer.borderColor = color.cgColor
        bgView.layer.borderWidth = 4
        bgView.backgroundColor = UIColor.clear
        addSubview(bgView)
    }
}

extension VNRecognizedObjectObservation {
    var label: String? {
        return self.labels.first?.identifier
    }
}
