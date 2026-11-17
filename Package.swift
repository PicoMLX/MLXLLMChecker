// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MLXLLMChecker",
    platforms: [.macOS(.v15), .iOS(.v18)],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "MLXLLMChecker",
            targets: ["MLXLLMChecker"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-huggingface", "0.8.1"..<"2.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "MLXLLMChecker",
            dependencies: [
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
        .testTarget(
            name: "MLXLLMCheckerTests",
            dependencies: ["MLXLLMChecker"]
        ),
    ]
)
