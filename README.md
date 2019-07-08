<p align="center">
  <img src="images/logo.png">
</p>

# Swift for TensorFlow

> TensorFlow를 위한 Swift: 경계선이 없다.

TensorFlow를 위한 Swift는 머신러닝, 컴파일러, Differentiable 프로그래밍, 시스템 디자인, 
그 이상의 것들에 대한 가장 최신 연구 내용을 결합시키는 머신러닝을 위한 차세대 플랫폼 입니다.
이것은 초기 단계의 프로젝트 입니다: 피처가 완전하지 않고, 상용화가 준비되지 않았지만, 선구자들로 하여금
프로젝트를 시도해볼 수 있는 수준입니다. 따라서, 피드백을 주시고 미래를 만들어나가는데 도움을 주세요!

TensorFlow를 위한 Swift 프로젝트는 현재 두 종류의 사용자를 대상으로 합니다:

1. **고급 머신러닝 연구자** 로, 현재의 머신러닝 프레임워크들에 한계를 느끼는 분들 입니다. 
   TensorFlow를 위한 Swift는 현대적인 다목적 언어와의 매끄러운 통합을 통하여
   보다 동적이면서도 보다 섬세한 모델을 만들 수 있다는 이점을 가지고 있습니다.
   빠른 추상화는 "사용자-공간에서"(C/C++의 "프레임워크-공간"과 반대 개념) 개발될 수 있는데,
   이는 손쉽게 사용자가 정의 가능한 모듈식의 APIs라느 결과를 가져오게 됩니다.

2. **머신러닝 학습자** 로, 머신러닝을 이제 막 시작하신 분들 입니다. Swift의 
   품질 도구 (컨텍스트-인지 자동완성)으 지원 덕분에, TensorFlow를 위한 Swift는
   머신러징의 기초를 배우는 시작점으로써 가장 생산적인 방법 중 하나일 수 있습니다.

## 시작 해 보기

### TensorFlow를 위한 Swift를 사용해 보기

- **Google Colaboratory**: TensorFlow를 위한 Swift를 여러분의 브러우저에 시도 해 보기 위한 가장 빠른 방법 입니다.
  단순히 [튜토리얼](#tutorials-) 를 열거나, [빈
   노트북](https://colab.research.google.com/github/tensorflow/swift/blob/master/notebooks/blank_swift.ipynb)!로 시작해 보시기 바랍니다. 더 많은 내용은 [사용법 가이드](Usage.md)을 읽어 보시기 바랍니다.

- **로컬 환경에 설치하는 방법**: [미리 빌드된, TensorFlow를 위한 Swift 패키지를 다운로드](Installation.md) 하실 수 있습니다.
  설치 완료 후, [단계별 설명](Usage.md)을 따라하셔서 Swift 스크립트를 여러분의 컴퓨터에서 빌드하고 실행해 보실 수 있습니다.

- **소스파일을 직접 컴파일 하는 방법**: TensorFlow를 위한 Swift를 커스터마이징 하거나
  어떤 기여를 하고 싶으시다면, TensorFlow를 위한 Swift를 소스파일로부터 빌드하기 위한 [지침](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow)으 따라 주시기 바랍니다.

### 튜토리얼 목록 ![](https://www.tensorflow.org/images/colab_logo_32px.png)

튜토리얼 | 마지막 업데이트 시간 |
-------- | ------------ |
[Swift 투어](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/a_swift_tour.ipynb) | 2019년 3월
[Python과의 상호 운용](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/python_interoperability.ipynb) | 2019년 3월
[사용자가 정의하는 Differentiation](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/custom_differentiation.ipynb) | 2019년 3월
[모델 학습의 단계적 설명](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb) | 2019년 3월
[가공되지 않은 TensorFlow 연산자들](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/raw_tensorflow_operators.ipynb) | 2019년 3월

### 리소스 목록

- [모델과 예제](https://github.com/tensorflow/swift-models)
- [TensorFlow의 Swift API 레퍼런스](https://www.tensorflow.org/api_docs/swift/Structs/Tensor)
- [릴리즈 노트](RELEASES.md)
- [알려진 문제점들](KNOWN_ISSUES.md)
- [자주 물어보는 질문들](FAQ.md)

### 포럼

부디 [swift@tensorflow.org 메일링 리스트](https://groups.google.com/a/tensorflow.org/d/forum/swift)에 참여하셔서, 
가장 최신의 발표를 확인하시고, 도움도 받으시고, 여러분의 생각도 공유해 주시기 바랍니다.

## 왜 TensorFlow를 위한 Swift 인가?

TensorFlow를 위한 Swift는 머신러닝 모델을 개발하기 위한 새로운 방법 입니다. 이것은 
[TensorFlow](https://www.tensorflow.org)의 강력함을 직접적으로
[Swift 프로그래밍 언어](https://swift.org/about)에 통합시켜 줍니다.
저희는 머신러닝의 패러다임이 매우 중요하기 때문에, **퍼스트-클래스 언어와 컴파일러의 지원** 이 마땅한 일이라고 믿고 있습니다.

머신러닝의 기본적인 요소는 경사도에 기반한 최적화로, 파라메터들을 최적화 하기 위한 함수의 미분을 계산하는 것입니다.
TensorFlow를 위한 Swift를 사용하면, [`gradient(of:)`](https://www.tensorflow.org/swift/api_docs/Functions#/s:10TensorFlow8gradient2of15CotangentVectorQzxcq_xc_tAA14DifferentiableRzSFR_AaFR_AdaFPQy_Rs_r0_lF) 와 같은 미분 연산자를 사용해서 손쉽게 함수를 미분하거나, 
모델의 [`gradient(in:)`](https://www.tensorflow.org/swift/api_docs/Protocols/Differentiable#/s:10TensorFlow14DifferentiablePAAE8gradient2in15CotangentVectorQzqd__xXE_tSFRd__AaBRd__AfCQyd__Rsd__lF) 메소드를 호출하여 모델 전체를 미분하는 것이 가능 합니다.
이러한 미분 APIs는 `Tensor`-에 연관된 개념- 에서는 사용이 불가능 하지만, `Float`, `Double`, SIMD 벡터, 여러분이 직접 만든 데이터 구조를 포함하여 [`Differentiable`](https://www.tensorflow.org/swift/api_docs/Protocols/Differentiable) 라는 프로토콜의 형식을 따르는 모든 데이터 타입들로 일반화 되어 있습니다.

```swift
// 사용자 정의 미분 가능한 데이터 타입.
struct Model: Differentiable {
    var w: Float
    var b: Float
    func applied(to input: Float) -> Float {
        return w * input + b
    }
}

// `Differentiable.gradient(at:in:)` 를 사용하여 미분 하기.
let model = Model(w: 4.0, b: 3.0)
let (𝛁model, 𝛁input) = model.gradient(at: 2.0) { model, input in
    model.applied(to: input)
}

print(𝛁model) // Model.AllDifferentiableVariables(w: 2.0, b: 1.0)
print(𝛁input) // 4.0
```

Beyond derivatives, the Swift for TensorFlow project comes with a sophisticated toolchain
to make users more productive. You can run Swift interactively in a Jupyter
notebook, and get helpful autocomplete suggestions to help you explore the
massive API surface of a modern deep learning library. You can [get started
right in your browser in
seconds](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb)!

Migrating to Swift for TensorFlow is really easy thanks to Swift's powerful
Python integration. You can incrementally migrate your Python code over (or
continue to use your favorite Python libraries), because you can easily call
your favorite Python library with a familiar syntax:

```swift
import TensorFlow
import Python

let np = Python.import("numpy")

let array = np.arange(100).reshape(10, 10)  // Create a 10x10 numpy array.
let tensor = Tensor<Float>(numpy: array)  // Seamless integration!
```











Swift for TensorFlow is a next-generation platform for machine learning, 
incorporating the latest research across machine learning, compilers, 
differentiable programming, systems design, and beyond. This is an 
early-stage project: it is not feature-complete nor production-ready, 
but it is ready for pioneers to try in projects, give feedback, and help shape the future!

The Swift for TensorFlow project is currently focusing on 2 kinds of users:

1. **Advanced ML researchers** who are limited by current ML frameworks. Swift
   for TensorFlow's advantages include a seamless integration with a modern
   general-purpose language, allowing for more dynamic and sophisticated models.
   Fast abstractions can be developed "in user-space" (as opposed to in C/C++
   aka "framework-space"), resulting in modular APIs that can be easily
   customized.

2. **ML learners** who are just getting started with machine learning. Thanks to
   Swift's support for quality tooling (e.g. context-aware autocomplete), Swift
   for TensorFlow can be one of the most productive ways to get started learning
   the fundamentals of machine learning.

## Getting started

### Using Swift for TensorFlow

- **Google Colaboratory**: The fastest way to get started is to try out Swift
   for TensorFlow right in your browser. Just open up [a tutorial](#tutorials-), or start from a [blank
   notebook](https://colab.research.google.com/github/tensorflow/swift/blob/master/notebooks/blank_swift.ipynb)!
   Read more in our [usage guide](Usage.md).

- **Install locally**: you can [download a pre-built Swift for TensorFlow
   package](Installation.md). After installation, you can follow these
   [step-by-step instructions](Usage.md) to build and execute a Swift script on
   your computer.

- **Compile from source**: If you'd like to customize Swift for TensorFlow or
   contribute back, follow our [instructions](https://github.com/apple/swift/tree/tensorflow#building-swift-for-tensorflow)
   on building Swift for TensorFlow from source.

### Tutorials ![](https://www.tensorflow.org/images/colab_logo_32px.png)

Tutorial | Last Updated |
-------- | ------------ |
[A Swift Tour](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/a_swift_tour.ipynb) | March 2019
[Python Interoperability](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/python_interoperability.ipynb) | March 2019
[Custom Differentiation](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/custom_differentiation.ipynb) | March 2019
[Model Training Walkthrough](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb) | March 2019
[Raw TensorFlow Operators](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/raw_tensorflow_operators.ipynb) | March 2019

### Resources

- [Models and Examples](https://github.com/tensorflow/swift-models)
- [TensorFlow Swift API Reference](https://www.tensorflow.org/api_docs/swift/Structs/Tensor)
- [Release Notes](RELEASES.md)
- [Known Issues](KNOWN_ISSUES.md)
- [Frequently Asked Questions](FAQ.md)

### Forums

Please join the
[swift@tensorflow.org mailing list](https://groups.google.com/a/tensorflow.org/d/forum/swift)
to hear the latest announcements, get help, and share your thoughts!

## Why Swift for TensorFlow?

Swift for TensorFlow is a new way to develop machine learning models. It
gives you the power of
[TensorFlow](https://www.tensorflow.org) directly integrated into the
[Swift programming language](https://swift.org/about). We believe that
machine learning paradigms are so important that they deserve
**first-class language and compiler support**.

A fundamental primitive in machine learning is gradient-based optimization:
computing function derivatives to optimize parameters. With Swift for
TensorFlow, you can easily differentiate functions using differential
operators like [`gradient(of:)`](https://www.tensorflow.org/swift/api_docs/Functions#/s:10TensorFlow8gradient2of15CotangentVectorQzxcq_xc_tAA14DifferentiableRzSFR_AaFR_AdaFPQy_Rs_r0_lF), or differentiate with respect to an entire
model by calling method [`gradient(in:)`](https://www.tensorflow.org/swift/api_docs/Protocols/Differentiable#/s:10TensorFlow14DifferentiablePAAE8gradient2in15CotangentVectorQzqd__xXE_tSFRd__AaBRd__AfCQyd__Rsd__lF). These differentiation APIs
are not just available for `Tensor`-related concepts—they are
generalized for all types that conform to the [`Differentiable`](https://www.tensorflow.org/swift/api_docs/Protocols/Differentiable)
protocol, including `Float`, `Double`, SIMD vectors, and your own data
structures.

```swift
// Custom differentiable type.
struct Model: Differentiable {
    var w: Float
    var b: Float
    func applied(to input: Float) -> Float {
        return w * input + b
    }
}

// Differentiate using `Differentiable.gradient(at:in:)`.
let model = Model(w: 4.0, b: 3.0)
let (𝛁model, 𝛁input) = model.gradient(at: 2.0) { model, input in
    model.applied(to: input)
}

print(𝛁model) // Model.AllDifferentiableVariables(w: 2.0, b: 1.0)
print(𝛁input) // 4.0
```

Beyond derivatives, the Swift for TensorFlow project comes with a sophisticated toolchain
to make users more productive. You can run Swift interactively in a Jupyter
notebook, and get helpful autocomplete suggestions to help you explore the
massive API surface of a modern deep learning library. You can [get started
right in your browser in
seconds](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb)!

Migrating to Swift for TensorFlow is really easy thanks to Swift's powerful
Python integration. You can incrementally migrate your Python code over (or
continue to use your favorite Python libraries), because you can easily call
your favorite Python library with a familiar syntax:

```swift
import TensorFlow
import Python

let np = Python.import("numpy")

let array = np.arange(100).reshape(10, 10)  // Create a 10x10 numpy array.
let tensor = Tensor<Float>(numpy: array)  // Seamless integration!
```

## Documentation

> Beware: the project is moving very quickly, and thus some of these documents
> are slightly out of date as compared to the current state-of-the-art.

### Overview

Document | Last Updated | Status |
-------- | ------------ | ------ |
[Why *Swift* for TensorFlow?](docs/WhySwiftForTensorFlow.md) | April 2018 | Current
[Swift for TensorFlow Design Overview](docs/DesignOverview.md) | April 2018 | Outdated

### Technology deep dive

The Swift for TensorFlow project builds on top of powerful theoretical
foundations. For insight into some of the underlying technologies, check
out the following documentation.

Document | Last Updated | Status |
-------- | ------------ | ------ |
[Swift Differentiable Programming Design Overview](https://docs.google.com/document/d/1bPepWLfRQa6CtXqKA8CDQ87uZHixNav-TFjLSisuKag/edit?usp=sharing) | June 2019 | Current
[Differentiable Types](docs/DifferentiableTypes.md) | March 2019 | Outdated
[Differentiable Functions and Differentiation APIs](docs/DifferentiableFunctions.md) | March 2019 | Outdated
[Dynamic Property Iteration using Key Paths](docs/DynamicPropertyIteration.md) | March 2019 | Current
[Hierarchical Parameter Iteration and Optimization](docs/ParameterOptimization.md) | March 2019 | Current
[First-Class Automatic Differentiation in Swift: A Manifesto](https://gist.github.com/rxwei/30ba75ce092ab3b0dce4bde1fc2c9f1d) | October 2018 | Outdated
[Automatic Differentiation Whitepaper](docs/AutomaticDifferentiation.md) | April 2018 | Outdated
[Python Interoperability](docs/PythonInteroperability.md) | April 2018 | Current
[Graph Program Extraction](docs/GraphProgramExtraction.md) | April 2018 | Outdated

## Source code

Compiler and standard library development happens on the `tensorflow` branch of
the [apple/swift](https://github.com/apple/swift/tree/tensorflow) repository.

Additional code repositories that make up the core of the project include:

 - [Swift fork of LLDB](http://github.com/apple/swift-lldb/tree/tensorflow):
   debugger and REPL support.
 - [Deep learning library](https://github.com/tensorflow/swift-apis): high-level
   API familiar to Keras users.

> Swift for TensorFlow is **not** intended to remain a long-term fork of the official
> Swift language. Language additions are designed to fit with the direction of
> Swift and will go through the [Swift
> Evolution](https://github.com/apple/swift-evolution) process.

### Jupyter Notebook support

[Jupyter Notebook](http://jupyter.org/) support for Swift is under development at
[google/swift-jupyter](https://github.com/google/swift-jupyter).

## Community

Swift for TensorFlow discussions happen on the
[swift@tensorflow.org mailing list](https://groups.google.com/a/tensorflow.org/d/forum/swift).

### Bugs reports and feature requests

Before reporting an issue, please check the [Frequently Asked Questions](FAQ.md)
to see if your question has already been addressed.

For questions about general use or feature requests, please send an email to
the [mailing list](mailto:swift@tensorflow.org) or search for relevant issues
in the [JIRA issue tracker](https://bugs.swift.org/projects/TF/issues/?filter=allopenissues).

For the most part, the core team's development is also tracked in
[JIRA](https://bugs.swift.org/secure/RapidBoard.jspa?rapidView=17&projectKey=TF&view=planning).

### Contributing

We welcome contributions from everyone. Read the [contributing
guide](Contributing.md) for information on how to get started.

### Code of conduct

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.
