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
// 사용자 정의 미분 가능한 데이터 타입 입니다..
struct Model: Differentiable {
    var w: Float
    var b: Float
    func applied(to input: Float) -> Float {
        return w * input + b
    }
}

// `Differentiable.gradient(at:in:)` 를 사용하여 미분이 가능합니다.
let model = Model(w: 4.0, b: 3.0)
let (𝛁model, 𝛁input) = model.gradient(at: 2.0) { model, input in
    model.applied(to: input)
}

print(𝛁model) // Model.AllDifferentiableVariables(w: 2.0, b: 1.0)
print(𝛁input) // 4.0
```

미분 이외에도, TensorFlow를 위한 Swift 프로젝트는 사용자를 더욱 생산적으로 만들어주는 섬세한 툴체인을 제공합니다.
여러분은 Jupyter Notebook에서 Swift를 대화식으로 실행해 보고, 현대의 방대한 딥러닝 라이브러리의 API를 탐구하는데 도움이 되는 자동완성 기능의 사용이 가능합니다. [여러분의 브라우저에서 빠르게, 바로 시작하기](https://colab.research.google.com/github/tensorflow/swift/blob/master/docs/site/tutorials/model_training_walkthrough.ipynb)! 를 해 보실 수 있습니다.

Swift의 강력한 Python 통합 덕분에, TensorFlow를 위한 Swift로 옮겨오는 것은 정말로 쉽습니다. 여러분이 이미 익숙한 문법으로 쓰여진 Python 라이브러리를 손쉽게 호출할 수 있기 때문에, 점진적으로 Python 코드를 옮겨오는 것이 가능합니다 (또는 Python 라이브러리를 있는 그대로 계속 사용할 수도 있습니다):

```swift
import TensorFlow
import Python

let np = Python.import("numpy")

let array = np.arange(100).reshape(10, 10)  // 10x10 numpy 배열을 생성 합니다.
let tensor = Tensor<Float>(numpy: array)  // 매끄러운 통합이 가능합니다!
```
## 문서

> 알아 두십시오: 프로젝트는 매우 빠르게 성장하고 있으므로, 이 문서의 몇 부분은
> 최신의 내용과 비교하면 약간 뒤쳐져 있을 수 있습니다.

### 오버뷰

문서 | 마지막 업데이트 날짜 | 상태 |
-------- | ------------ | ------ |
[왜 TensorFlow를 위한 *Swift* 인가?](docs/WhySwiftForTensorFlow.md) | 2018년 4월 | 최신
[TensorFlow를 위한 Swift에 대한 디자인 오버뷰](docs/DesignOverview.md) | 2018년 4월 | 구식

### 기술의 깊은 이해

TensorFlow를 위한 Swift 프로젝트는 강력한 이론적 기초를 기반으로 만들어 졌습니다.
그 기반이 되는 몇 가지 기술들에 대한 이해를 위해서, 아래의 문서를 확인해 보시기 바랍니다.

문서 | 마지막 업데이트 날짜 | 상태 |
-------- | ------------ | ------ |
[Swift의 Differentiable 프로그래밍 디자인 오버뷰](https://docs.google.com/document/d/1bPepWLfRQa6CtXqKA8CDQ87uZHixNav-TFjLSisuKag/edit?usp=sharing) | 2019년 6월 | 최신
[미분 가능한 데이터 타입](docs/DifferentiableTypes.md) | 2019년 3월 | 구식
[미분 가능한 함수와 APIs](docs/DifferentiableFunctions.md) | 2019년 3월 | 구식
[Key Paths를 이용한 동적 속성 순회](docs/DynamicPropertyIteration.md) | 2019년 3월 | 최신
[계층적 파라메터 순회 및 최적화](docs/ParameterOptimization.md) | 2019년 3월 | 최신
[Swift에서의 프스트-클래스 자동 미분: A Manifesto](https://gist.github.com/rxwei/30ba75ce092ab3b0dce4bde1fc2c9f1d) | 2018년 10월 | 구식
[자동 미분 백서](docs/AutomaticDifferentiation.md) | 2018년 4월 | 구식
[Python 과의 상호 운용](docs/PythonInteroperability.md) | 2018년 4월 | 최신
[그래프 프로그램 추출](docs/GraphProgramExtraction.md) | 2018년 4월 | 구식

## 소스 코드

컴파일러 및 표준 라이브러리의 개발은 [apple/swift](https://github.com/apple/swift/tree/tensorflow) 저장소의 `tensorflow` 브랜치에서 진행 중 입니다.

프로젝트의 핵심을 담고 있는 추가적인 코드 저장소로는 다음이 포함 됩니다.
- [LLDB에서 갈라져나온 Swift](http://github.com/apple/swift-lldb/tree/tensorflow):
   디버거 및 REPL에 관련된 저장소 입니다.
 - [딥러닝 라이브러리](https://github.com/tensorflow/swift-apis): Keras 사용자에게 친숙한 고수준 API에 관련된 저장소 입니다.

> TensorFlow를 위한 장기적으로 Swift는 공식 Swift 언어의 갈래로 떨어져나오길 
> 의도하지 *않습니다*. 언어의 추가 요소는 Swift가 나아가는 방향성과 들어맞기 위해
> 디자인 되었으며, 그 방향은 [Swift의 진화](https://github.com/apple/swift-evolution) 프로세스에서 확인해 보실 수 있습니다.

### Jupyter Notebook 지원

Swift를 위한 [Jupyter Notebook](http://jupyter.org/) 지원은 [google/swift-jupyter](https://github.com/google/swift-jupyter) 에서 현재 개발 중 입니다.

## 커뮤니티

TensorFlow를 위한 Swift에 대한 논의는 를 위한 [swift@tensorflow.org mailing list](https://groups.google.com/a/tensorflow.org/d/forum/swift)를 통해서 이뤄지고 있습니다.

### 버그 리포트 및 추가적인 피처에 대한 요청

이슈를 리포트 하기 전에, [자주 물어보는 질문](FAQ.md)을 확인해 보시고 
여쭤보려고 하시는 질문의 답이 이미 있는지 확인해 보시기 바랍니다.

일반적인 사용법 또는 추가 피쳐에 대한 요청에 대해서는 [메일링 리스트](mailto:swift@tensorflow.org)로 이메일을 보내거나
[JIRA issue tracker](https://bugs.swift.org/projects/TF/issues/?filter=allopenissues) 에서 관련된 이슈를 검색해 보시기 바랍니다.

대부분의 경우에서, 핵심 팀의 개발 또한 [JIRA](https://bugs.swift.org/secure/RapidBoard.jspa?rapidView=17&projectKey=TF&view=planning) 에서 추적되고 있음을 알려 드립니다.

### 기여

여러분 모두로부터의 기여를 환영합니다. 이에 대한 더 많은 정보 및 시작하는 방법은 [contributing
guide](Contributing.md)를 확인해 보시기 바랍니다.

### 행동 규칙



In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of
experience, education, socio-economic status, nationality, personal appearance,
race, religion, or sexual identity and orientation.

The Swift for TensorFlow community is guided by our [Code of
Conduct](CODE_OF_CONDUCT.md), which we encourage everybody to read before
participating.
