//
//  ViewController.swift
//  DeepLearning
//
//  Created by Brad Hesse on 9/5/17.
//  Copyright © 2017 Brad Hesse. All rights reserved.
//

import UIKit
import AVFoundation
import CoreML
import Vision

class ViewController: UIViewController
{
    // MARK: IBOutlets
    @IBOutlet weak var previewContainerView: UIView!
    @IBOutlet weak var topLayoutConstraint: NSLayoutConstraint!
    @IBOutlet weak var bottomLayoutConstraint: NSLayoutConstraint!
    @IBOutlet weak var observationLabel: UILabel!
    
    //Camera Capture required properties
    var videoDataOutput: AVCaptureVideoDataOutput!
    var videoDataOutputQueue: DispatchQueue!
    var previewLayer:AVCaptureVideoPreviewLayer!
    var captureDevice : AVCaptureDevice!
    let session = AVCaptureSession()
    var deviceInut : AVCaptureDeviceInput!;
    
    //CoreML properties.
    var visionRequests = [VNRequest]();
    let labels = ["apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"];
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        self.captureDevice = AVCaptureDevice.default(for: .video);
        self.session.sessionPreset = AVCaptureSession.Preset.high;
        
        self.beginSession();
        
        NotificationCenter.default.addObserver(self, selector: #selector(self.didAutoRotate), name: NSNotification.Name.UIDeviceOrientationDidChange, object: nil);
        
        guard let visionModel = try? VNCoreMLModel(for: tensormodel().model) else { fatalError("could not instantiate tensormodel") };
        
        let classificationRequest = VNCoreMLRequest(model: visionModel) { (request, error) in
            //create the classification request, handle the response.
            if let error = error {
                print("ENCOUNTERED ERROR: \(error)");
                return;
            }
            
            guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
                if let res = request.results {
                    for obj in res {
                        print(obj);
                    }
                }
                return;
            }
            
            if observations.first?.featureValue.type == .multiArray {
                //look for the output with the highest value.
                //this index will correspond with a label in our self.labels array
                
                guard let res = observations.first?.featureValue.multiArrayValue else {
                    return;
                }
                
                guard (res.count == 100) else {
                    return;
                }
                
                var highestIndex = 0;
                
                for i in 0...99 {
                    let val = res[i];
                    
                    if (val.doubleValue > res[highestIndex].doubleValue) {
                        highestIndex = i;
                    }
                }
                
                DispatchQueue.main.async {
                    if (res[highestIndex].doubleValue > 0.50) {
                        self.observationLabel.text = "\(self.labels[highestIndex]) (\(String(format: "%.f", res[highestIndex].doubleValue * 100.0))%)";
                    } else {
                        self.observationLabel.text = "None";
                    }
                };
            }
        };
        
        classificationRequest.imageCropAndScaleOption = .centerCrop;
        
        self.visionRequests = [classificationRequest];
    }
    
    @objc func didAutoRotate()
    {
        //when the device rotates, this function sets the 'blur' bars to be either on top and bottom (portrait) or side to size (landscape) so that the image the user sees is always a square image.
        if (UIDevice.current.orientation == .landscapeLeft || UIDevice.current.orientation == .landscapeRight) {
            self.topLayoutConstraint.priority = UILayoutPriority(rawValue: 910);
            self.bottomLayoutConstraint.priority = UILayoutPriority(rawValue: 910);
            self.previewLayer.connection?.videoOrientation = (UIDevice.current.orientation == .landscapeRight) ? AVCaptureVideoOrientation.landscapeLeft : AVCaptureVideoOrientation.landscapeRight;
            
        } else if (UIDevice.current.orientation == .portrait || UIDevice.current.orientation == .portraitUpsideDown) {
            self.topLayoutConstraint.priority = UILayoutPriority(rawValue: 890);
            self.bottomLayoutConstraint.priority = UILayoutPriority(rawValue: 890);
            self.previewLayer.connection?.videoOrientation = (UIDevice.current.orientation == .portrait) ? AVCaptureVideoOrientation.portrait : AVCaptureVideoOrientation.portraitUpsideDown;
        }
        
        self.view.layoutIfNeeded();
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }
    
    override func viewDidAppear(_ animated: Bool)
    {
        self.previewLayer.frame = self.view.bounds;
    }
    
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator)
    {
        //handles rotations and adjusts size of the previewLayer accordingly
        coordinator.animate(alongsideTransition: { (context) in
            self.previewLayer.frame = self.view.bounds;
        }, completion: nil);
    }
    
    func beginSession()
    {
        //begins the AVCaptureSession session
        
        self.deviceInut = try! AVCaptureDeviceInput(device: self.captureDevice);
        
        guard (self.session.canAddInput(self.deviceInut)) else { return };
        
        self.session.addInput(self.deviceInut);
        
        self.videoDataOutput = AVCaptureVideoDataOutput();
        self.videoDataOutput.alwaysDiscardsLateVideoFrames = true;
        self.videoDataOutputQueue = DispatchQueue(label: "videoQueue");
        self.videoDataOutput.setSampleBufferDelegate(self, queue: self.videoDataOutputQueue);
        print("DID SET SAMPLE BUFFER DELEGATE");
        guard (self.session.canAddOutput(self.videoDataOutput)) else { return };
        
        self.session.addOutput(self.videoDataOutput);
        
        self.videoDataOutput.connection(with: .video)?.isEnabled = true;
        
        self.previewLayer = AVCaptureVideoPreviewLayer(session: self.session);
        
        self.previewLayer.videoGravity = .resizeAspectFill;
        
        let rootLayer = self.previewContainerView.layer;
        rootLayer.masksToBounds = true;
        self.previewLayer.bounds = self.view.bounds;
        rootLayer.addSublayer(self.previewLayer);
        self.session.startRunning();
    }
}

extension ViewController:  AVCaptureVideoDataOutputSampleBufferDelegate
{
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection)
    {
        //this delegate function captures camera output, converts it to a pixel buffer, and feeds it into the image request handler, and performs inference on that input
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return;
        }
        
        var requestOptions = [VNImageOption : Any]();
        
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics : cameraIntrinsicData];
        }
        
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: CGImagePropertyOrientation(rawValue: UInt32(Int32(UIDevice.current.orientation.rawValue)))!, options: requestOptions)
        
        do {
            try imageRequestHandler.perform(self.visionRequests);
            
            
        } catch let error {
            print("ENCOUNTERED ERROR 2: \(error)");
        }
    }
}
