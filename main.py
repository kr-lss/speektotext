"""
===============================================================================
실시간 토론 분석 시스템 - Google Cloud API 직접 호출 데모
===============================================================================

프로젝트 목적:
    친구들끼리 토론할 때 (예: "방망이든 오타니 3명 vs 사자 1마리")
    각자의 주장을 음성으로 듣고, AI가 논리와 근거를 분석하여 판단

사용 기술:
    1. Google Cloud Speech-to-Text API (음성→텍스트)
    2. Google Vertex AI Gemini API (텍스트 분석 및 판단)

코드 출처 및 참고:
    이 코드는 Google Cloud 공식 샘플을 기반으로 작성되었습니다.
===============================================================================
"""

import queue
import sys
import time
from google.cloud import speech_v2 as speech
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
import vertexai
from vertexai.generative_models import GenerativeModel
import pyaudio


# ========== 프로젝트 설정 ==========
PROJECT_ID = "knu-sungsu613"  # TODO: Google Cloud 프로젝트 ID 입력
LOCATION = "global"
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE / 10)  # 100ms


# ===============================================================================
# 1. MicrophoneStream 클래스
# ===============================================================================
# 출처: GoogleCloudPlatform/python-docs-samples
# 파일: speech/microphone/transcribe_streaming_mic.py
# URL: https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/speech/microphone/transcribe_streaming_mic.py
#
# 설명: 마이크로부터 실시간으로 오디오를 캡처하여 스트림으로 제공
#       pyaudio를 사용하여 마이크 입력을 queue에 저장하고 generator로 제공
# ===============================================================================

class MicrophoneStream:
    """마이크 스트림 관리 클래스
    
    Google Cloud 공식 샘플 코드 기반
    출처: python-docs-samples/speech/microphone/transcribe_streaming_mic.py
    """
    
    def __init__(self, rate=SAMPLE_RATE, chunk=CHUNK_SIZE):
        """마이크 스트림 초기화
        
        Args:
            rate: 샘플링 레이트 (기본 16000Hz)
            chunk: 오디오 청크 크기 (기본 100ms)
        """
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()  # 오디오 데이터를 저장할 큐
        self.closed = True
    
    def __enter__(self):
        """컨텍스트 매니저 진입 - 마이크 스트림 시작"""
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,      # 16비트 PCM 포맷
            channels=1,                   # 모노 오디오
            rate=self._rate,
            input=True,                   # 입력 모드
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,  # 콜백 함수
        )
        self.closed = False
        return self
    
    def __exit__(self, type, value, traceback):
        """컨텍스트 매니저 종료 - 마이크 스트림 정리"""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)  # 종료 신호
        self._audio_interface.terminate()
    
    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """오디오 콜백 함수 - 마이크 입력을 버퍼에 저장
        
        설명: pyaudio가 자동으로 호출하는 콜백 함수
              마이크로 들어오는 오디오 데이터를 queue에 저장
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue
    
    def generator(self):
        """오디오 청크 생성기
        
        설명: queue에서 오디오 데이터를 꺼내서 yield
              Speech-to-Text API로 전송할 데이터 스트림 생성
        """
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            
            # 버퍼에 쌓인 추가 데이터도 함께 가져오기
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            
            yield b"".join(data)


# ===============================================================================
# 2. DebateAnalyzer 클래스 (메인 분석 시스템)
# ===============================================================================
# 코드 구성:
#   - Speech-to-Text API 설정 부분: 
#     출처: GoogleCloudPlatform/generative-ai
#     파일: audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb
#     URL: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb
#
#   - Gemini 분석 부분:
#     출처: GoogleCloudPlatform/generative-ai
#     파일: gemini/prompts/examples/text_summarization.ipynb
#     URL: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/prompts/examples/text_summarization.ipynb
# ===============================================================================

class DebateAnalyzer:
    """실시간 토론 분석 시스템
    
    기능:
        1. 마이크로 음성 입력 받기
        2. Speech-to-Text로 실시간 변환
        3. Gemini로 주장의 논리와 근거 분석
    """
    
    def __init__(self, project_id, language_code="ko-KR"):
        """분석 시스템 초기화
        
        Args:
            project_id: Google Cloud 프로젝트 ID
            language_code: 언어 코드 (기본값: 한국어 "ko-KR")
        """
        self.project_id = project_id
        self.language_code = language_code
        self.transcript_buffer = []  # 인식된 텍스트 저장
        
        # Speech-to-Text 클라이언트 초기화
        print("🔧 Google Cloud Speech-to-Text API 초기화 중...")
        self.speech_client = SpeechClient()
        
        # Vertex AI Gemini 초기화
        print("🔧 Google Vertex AI Gemini 초기화 중...")
        vertexai.init(project=project_id, location="us-central1")
        self.gemini_model = GenerativeModel("gemini-1.5-flash")
        
        print("\n✅ 시스템 초기화 완료!")
        print(f"   📌 Speech-to-Text: {language_code} 언어 인식")
        print(f"   📌 Gemini Model: gemini-1.5-flash")
        print(f"   📌 Project ID: {project_id}\n")
    
    
    # ===========================================================================
    # Speech-to-Text API 설정
    # ===========================================================================
    # 출처: get_started_with_chirp_2_sdk_features.ipynb
    # 
    # 설명: Google Cloud Speech-to-Text V2 API 설정
    #       - RecognitionConfig: 음성 인식 설정 (언어, 모델 등)
    #       - StreamingRecognitionConfig: 스트리밍 설정 (실시간 인식)
    # ===========================================================================
    
    def create_stream_config(self):
        """Speech-to-Text 스트리밍 설정 생성
        
        Google Cloud 공식 샘플 기반
        출처: audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb
        
        Returns:
            StreamingRecognitionConfig: 스트리밍 인식 설정
        """
        
        # 1. Recognition Config 생성
        recognition_config = cloud_speech.RecognitionConfig(
            # 오디오 인코딩 자동 감지
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            
            # 인식할 언어 설정 (한국어: ko-KR, 영어: en-US)
            language_codes=[self.language_code],
            
            # 모델 선택: "long" = 긴 대화에 최적화된 모델
            # 다른 옵션: "short", "chirp_2", "chirp_3"
            model="long",
            
            # 인식 기능 설정
            features=cloud_speech.RecognitionFeatures(
                enable_automatic_punctuation=True,  # 자동 문장부호 추가
                enable_word_time_offsets=True,      # 단어별 타임스탬프
                enable_word_confidence=True,        # 단어별 신뢰도
            )
        )
        
        # 2. Streaming Config 생성
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True  # 중간 결과도 반환 (실시간 표시용)
            )
        )
        
        return streaming_config
    
    
    # ===========================================================================
    # 실시간 음성 인식 (Speech-to-Text API 직접 호출)
    # ===========================================================================
    # 출처: GoogleCloudPlatform/python-docs-samples
    # 파일: speech/snippets/transcribe_streaming_v2.py
    # URL: https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/speech/snippets/transcribe_streaming_v2.py
    #
    # 설명: 마이크 스트림을 Speech-to-Text API로 전송하여 실시간 변환
    # ===========================================================================
    
    def transcribe_streaming(self, audio_generator):
        """실시간 음성→텍스트 변환
        
        Google Cloud API 직접 호출
        출처: speech/snippets/transcribe_streaming_v2.py
        
        Args:
            audio_generator: 오디오 데이터 생성기 (MicrophoneStream.generator)
        """
        
        print("\n" + "="*70)
        print("🎤 실시간 음성 인식 시작")
        print("="*70)
        print("💡 토론 내용을 말씀해주세요. '종료'라고 말하면 분석을 시작합니다.")
        print("-"*70 + "\n")
        
        streaming_config = self.create_stream_config()
        
        # 첫 번째 요청: 설정 정보
        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=f"projects/{self.project_id}/locations/{LOCATION}/recognizers/_",
            streaming_config=streaming_config
        )
        
        def request_generator():
            """API 요청 생성기
            
            설명: 첫 요청은 설정, 이후 요청은 오디오 데이터
            """
            yield config_request
            for audio_chunk in audio_generator:
                yield cloud_speech.StreamingRecognizeRequest(audio=audio_chunk)
        
        # ★★★ Google Cloud Speech-to-Text API 직접 호출 ★★★
        responses = self.speech_client.streaming_recognize(
            requests=request_generator()
        )
        
        # 응답 처리
        print("인식 결과:")
        print("-"*70)
        
        for response in responses:
            if not response.results:
                continue
            
            result = response.results[0]
            if not result.alternatives:
                continue
            
            transcript = result.alternatives[0].transcript
            is_final = result.is_final
            confidence = result.alternatives[0].confidence if hasattr(result.alternatives[0], 'confidence') else 0
            
            # 중간 결과 표시 (회색으로 표시)
            if not is_final:
                print(f"\r⏳ [인식 중] {transcript}                    ", end="", flush=True)
            else:
                # 최종 결과 (확정된 텍스트)
                print(f"\r✅ [확정] {transcript} (신뢰도: {confidence:.1%})")
                self.transcript_buffer.append(transcript)
                
                # 종료 명령 체크
                if "종료" in transcript or "quit" in transcript.lower():
                    print("\n" + "-"*70)
                    print("🛑 음성 인식 종료\n")
                    return
    
    
    # ===========================================================================
    # Gemini를 이용한 토론 분석
    # ===========================================================================
    # 출처: GoogleCloudPlatform/generative-ai
    # 파일: gemini/prompts/examples/text_summarization.ipynb
    # URL: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/prompts/examples/text_summarization.ipynb
    #
    # 설명: Gemini API를 사용하여 토론 내용의 논리와 근거 분석
    # ===========================================================================
    
    def analyze_debate(self):
        """Gemini로 토론 내용 분석
        
        Google Cloud Gemini API 직접 호출
        출처: gemini/prompts/examples/text_summarization.ipynb
        
        분석 내용:
            1. 주요 주장 파악
            2. 논리적 근거 평가
            3. 강점과 약점 분석
            4. 최종 판단
        """
        
        if not self.transcript_buffer:
            print("⚠️  분석할 토론 내용이 없습니다.")
            return
        
        # 전체 대화 내용 결합
        full_transcript = "\n".join([
            f"발언 {i+1}: {text}" 
            for i, text in enumerate(self.transcript_buffer)
        ])
        
        print("\n" + "="*70)
        print("🤖 Gemini AI 토론 분석 시작")
        print("="*70)
        print(f"📝 분석할 발언 수: {len(self.transcript_buffer)}개")
        print("-"*70 + "\n")
        
        # Gemini 프롬프트 작성
        prompt = f"""
당신은 토론 분석 전문가입니다. 아래 토론 내용을 객관적으로 분석해주세요.

## 📋 토론 내용:
{full_transcript}

## 🎯 분석 요청 사항:

### 1. 주요 주장 정리
각 발언자의 핵심 주장을 요약해주세요.

### 2. 논리적 근거 평가 (5점 만점)
- 각 주장의 논리성과 근거의 타당성을 평가
- 점수와 함께 이유를 설명

### 3. 강점과 약점
- 각 주장의 강점 (논리적으로 우수한 부분)
- 각 주장의 약점 (논리적 허점이나 근거 부족)

### 4. 최종 판단
- 어느 주장이 더 논리적이고 설득력 있는가?
- 그 이유는 무엇인가?

분석 결과를 명확하고 구조화된 형식으로 작성해주세요.
이모지를 적절히 사용하여 가독성을 높여주세요.
"""
        
        # ★★★ Google Vertex AI Gemini API 직접 호출 ★★★
        print("⏳ Gemini가 분석 중입니다...\n")
        response = self.gemini_model.generate_content(prompt)
        
        print("="*70)
        print("📊 AI 분석 결과")
        print("="*70)
        print(response.text)
        print("="*70 + "\n")
        
        return response.text


# ===============================================================================
# 메인 실행 함수
# ===============================================================================

def main():
    """메인 실행 함수
    
    실행 흐름:
        1. DebateAnalyzer 초기화 (API 클라이언트 설정)
        2. 마이크로부터 실시간 음성 입력
        3. Speech-to-Text로 텍스트 변환
        4. Gemini로 토론 분석 및 판단
    """
    
    print("\n" + "="*70)
    print("🎯 실시간 토론 분석 시스템")
    print("="*70)
    print("📌 예시 주제: 방망이든 오타니 3명 vs 사자 1마리, 누가 이길까?")
    print("="*70 + "\n")
    
    # 1. 분석 시스템 초기화
    analyzer = DebateAnalyzer(
        project_id=PROJECT_ID,
        language_code="ko-KR"  # 한국어 인식
    )
    
    # 2. 마이크로부터 실시간 음성 입력 및 변환
    try:
        with MicrophoneStream() as stream:
            audio_generator = stream.generator()
            analyzer.transcribe_streaming(audio_generator)
    except KeyboardInterrupt:
        print("\n⚠️  사용자가 중단했습니다.")
        return
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return
    
    # 3. Gemini로 토론 분석
    analyzer.analyze_debate()
    
    print("\n✅ 분석 완료!")


if __name__ == "__main__":
    """프로그램 시작점
    
    실행 전 확인사항:
        1. Google Cloud 프로젝트 생성
        2. Speech-to-Text API 활성화
        3. Vertex AI API 활성화
        4. 인증 설정 (gcloud auth application-default login)
        5. PROJECT_ID 변수 수정
        6. 필요한 라이브러리 설치:
           pip install google-cloud-speech
           pip install google-cloud-aiplatform
           pip install vertexai
           pip install pyaudio
    """
    main()


# ===============================================================================
# 코드 출처 요약
# ===============================================================================
"""
1. MicrophoneStream 클래스:
   - 출처: GoogleCloudPlatform/python-docs-samples
   - 파일: speech/microphone/transcribe_streaming_mic.py
   - 기능: 마이크 입력을 실시간으로 캡처

2. Speech-to-Text 설정 (create_stream_config):
   - 출처: GoogleCloudPlatform/generative-ai
   - 파일: audio/speech/getting-started/get_started_with_chirp_2_sdk_features.ipynb
   - 기능: 음성 인식 API 설정

3. 스트리밍 변환 (transcribe_streaming):
   - 출처: GoogleCloudPlatform/python-docs-samples
   - 파일: speech/snippets/transcribe_streaming_v2.py
   - 기능: 실시간 음성→텍스트 변환

4. Gemini 분석 (analyze_debate):
   - 출처: GoogleCloudPlatform/generative-ai
   - 파일: gemini/prompts/examples/text_summarization.ipynb
   - 기능: 텍스트 분석 및 요약

모든 코드는 Google Cloud 공식 샘플을 기반으로 작성되었으며,
프로젝트 목적에 맞게 통합 및 수정되었습니다.
"""
# ===============================================================================