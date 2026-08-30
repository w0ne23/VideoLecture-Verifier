"""
오디오 추출, 분석, 품질 평가
"""

import subprocess
import os
import wave

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None


def extract_audio_from_video(video_path: str, output_path: str = "temp_audio.wav") -> str:
    """영상에서 오디오 추출 (전체)"""
    print(f"  오디오 추출 중...")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-y", output_path
    ], capture_output=True)
    return output_path


def analyze_audio_features(audio_path: str) -> dict:
    """librosa로 오디오 피처 추출"""
    print(f"  오디오 로드 중...")
    if librosa is None:
        return _analyze_audio_features_basic(audio_path)

    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr

    print(f"  피처 추출 중... (길이: {duration/60:.1f}분)")

    features = {
        'metadata': {
            'duration_seconds': float(duration),
            'sample_rate': int(sr),
            'num_samples': len(y)
        }
    }

    # 1. MFCC (음성 지문)
    print("    - MFCC...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc'] = {
        'shape': list(mfcc.shape),
        'mean': mfcc.mean(axis=1).tolist(),
        'std': mfcc.std(axis=1).tolist(),
        'mean_avg': float(np.mean(mfcc.mean(axis=1))),
        'std_avg': float(np.mean(mfcc.std(axis=1))),
        'description': '음성 특징 - 목소리 고유 특성'
    }

    # 2. 멜 스펙트로그램 (주파수별 에너지)
    print("    - 멜 스펙트로그램...")
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_spectrogram'] = {
        'shape': list(mel_spec.shape),
        'mean_energy_db': float(np.mean(mel_spec_db)),
        'max_energy_db': float(np.max(mel_spec_db)),
        'min_energy_db': float(np.min(mel_spec_db)),
        'description': '주파수별 에너지 분포'
    }

    # 3. RMS 에너지 (볼륨)
    print("    - RMS 에너지...")
    rms = librosa.feature.rms(y=y)[0]
    features['rms_energy'] = {
        'mean': float(np.mean(rms)),
        'std': float(np.std(rms)),
        'max': float(np.max(rms)),
        'min': float(np.min(rms)),
        'description': '평균 볼륨/에너지'
    }

    # 4. 제로 크로싱 레이트 (음성 활동도)
    print("    - 제로 크로싱 레이트...")
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zero_crossing_rate'] = {
        'mean': float(np.mean(zcr)),
        'std': float(np.std(zcr)),
        'description': '음성 활동도 지표'
    }

    # 5. 스펙트럴 센트로이드 (소리의 밝기)
    print("    - 스펙트럴 센트로이드...")
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid'] = {
        'mean': float(np.mean(spectral_centroids)),
        'std': float(np.std(spectral_centroids)),
        'max': float(np.max(spectral_centroids)),
        'min': float(np.min(spectral_centroids)),
        'description': '음색의 밝기 (Hz) - 높을수록 선명함'
    }

    # 6. 스펙트럴 롤오프 (고주파 에너지)
    print("    - 스펙트럴 롤오프...")
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff'] = {
        'mean': float(np.mean(spectral_rolloff)),
        'std': float(np.std(spectral_rolloff)),
        'max': float(np.max(spectral_rolloff)),
        'min': float(np.min(spectral_rolloff)),
        'description': '고주파 에너지 분포 (Hz) - 음질 지표'
    }

    # 7. 템포 (말하기 속도)
    # librosa.beat.beat_track can crash the native process on some long lecture
    # inputs. It is non-critical for the pipeline, so keep it opt-in.
    tempo_flag = os.getenv("VLVERIFIER_AUDIO_TEMPO_ENABLED", "0").strip().lower()
    tempo_enabled = bool(tempo_flag) and tempo_flag not in {"0", "false", "no"}
    if tempo_enabled:
        print("    - 템포 추정...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = {
            'bpm': float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
            'description': '말하기 속도 지표 (BPM)'
        }
    else:
        print("    - 템포 추정 건너뜀...")
        features['tempo'] = {
            'bpm': None,
            'skipped': True,
            'description': '말하기 속도 지표 (비활성화됨)'
        }

    # 8. 침묵 구간 감지
    print("    - 침묵 구간 분석...")
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    threshold_db = -40
    is_silent = rms_db < threshold_db

    # 침묵 비율 계산
    silence_frames = np.sum(is_silent)
    total_frames = len(is_silent)
    silence_ratio = silence_frames / total_frames if total_frames > 0 else 0

    features['silence_detection'] = {
        'threshold_db': threshold_db,
        'silence_ratio': float(silence_ratio),
        'speech_ratio': float(1 - silence_ratio),
        'silence_seconds': float(duration * silence_ratio),
        'speech_seconds': float(duration * (1 - silence_ratio)),
        'description': '침묵 vs 음성 비율'
    }

    print(f"  ✓ 피처 추출 완료")
    return features


def _analyze_audio_features_basic(audio_path: str) -> dict:
    """Fallback audio summary when librosa is unavailable."""
    print("  librosa 없음: 기본 PCM 오디오 요약만 생성합니다.")
    with wave.open(audio_path, "rb") as wav:
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
        sample_width = wav.getsampwidth()
        channels = wav.getnchannels()

    if sample_width != 2:
        y = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        y = (y - np.mean(y)) / (np.max(np.abs(y - np.mean(y))) or 1.0)
    else:
        y = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1 and len(y) >= channels:
        y = y.reshape(-1, channels).mean(axis=1)

    duration = len(y) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
    silence_ratio = float(np.mean(np.abs(y) < 0.01)) if len(y) else 1.0

    return {
        "metadata": {
            "duration_seconds": float(duration),
            "sample_rate": int(sr),
            "num_samples": int(len(y)),
            "fallback": "basic_pcm_no_librosa",
        },
        "mfcc": {
            "shape": [0, 0],
            "mean": [],
            "std": [],
            "mean_avg": 0.0,
            "std_avg": 0.0,
            "description": "librosa unavailable",
        },
        "mel_spectrogram": {
            "shape": [0, 0],
            "mean_energy_db": 0.0,
            "max_energy_db": 0.0,
            "min_energy_db": 0.0,
            "description": "librosa unavailable",
        },
        "rms_energy": {
            "mean": rms,
            "std": 0.0,
            "max": rms,
            "min": rms,
            "description": "basic RMS energy",
        },
        "zero_crossing_rate": {
            "mean": 0.0,
            "std": 0.0,
            "description": "librosa unavailable",
        },
        "spectral_centroid": {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": 0.0,
            "description": "librosa unavailable",
        },
        "spectral_rolloff": {
            "mean": 0.0,
            "std": 0.0,
            "max": 0.0,
            "min": 0.0,
            "description": "librosa unavailable",
        },
        "tempo": {
            "bpm": None,
            "skipped": True,
            "description": "librosa unavailable",
        },
        "silence_detection": {
            "threshold_db": None,
            "silence_ratio": silence_ratio,
            "speech_ratio": float(1 - silence_ratio),
            "silence_seconds": float(duration * silence_ratio),
            "speech_seconds": float(duration * (1 - silence_ratio)),
            "description": "basic amplitude threshold silence estimate",
        },
    }


def evaluate_audio_quality(features: dict) -> dict:
    """오디오 피처 기반 강의 품질 자동 평가"""
    evaluation = {
        'overall_score': 0,
        'total_checks': 0,
        'passed_checks': 0,
        'details': {}
    }

    # 평가 기준 (하드코딩)
    criteria = {
        'rms_energy': {
            'name': '볼륨 크기',
            'value': features['rms_energy']['mean'],
            'min': 0.05,
            'max': 0.5,
            'optimal': (0.15, 0.35),
            'unit': '',
            'description': '너무 작거나 크지 않은 적절한 볼륨'
        },
        'spectral_centroid': {
            'name': '발음 선명도',
            'value': features['spectral_centroid']['mean'],
            'min': 800,
            'max': 3000,
            'optimal': (1200, 2000),
            'unit': 'Hz',
            'description': '명확하고 또렷한 발음'
        },
        'spectral_rolloff': {
            'name': '녹음 품질',
            'value': features['spectral_rolloff']['mean'],
            'min': 1500,
            'max': 6000,
            'optimal': (2000, 4000),
            'unit': 'Hz',
            'description': '고품질 녹음 환경'
        },
        'silence_ratio': {
            'name': '침묵 비율',
            'value': features['silence_detection']['silence_ratio'] * 100,
            'min': 5,
            'max': 35,
            'optimal': (10, 25),
            'unit': '%',
            'description': '적절한 쉬는 시간'
        },
        'mfcc_stability': {
            'name': '음성 안정성',
            'value': features['mfcc']['std_avg'],
            'min': 0,
            'max': 50,
            'optimal': (10, 30),
            'unit': '',
            'description': '깨끗하고 안정적인 녹음'
        }
    }
    tempo_value = features.get('tempo', {}).get('bpm')
    if isinstance(tempo_value, (int, float)) and np.isfinite(tempo_value):
        criteria['tempo'] = {
            'name': '말하기 속도',
            'value': tempo_value,
            'min': 80,
            'max': 150,
            'optimal': (100, 130),
            'unit': 'BPM',
            'description': '이해하기 적절한 속도'
        }

    # 각 항목 평가
    for key, criterion in criteria.items():
        evaluation['total_checks'] += 1

        value = criterion['value']
        is_passed = criterion['min'] <= value <= criterion['max']
        is_optimal = criterion['optimal'][0] <= value <= criterion['optimal'][1]

        # 상태 판단
        if is_optimal:
            status = '우수'
            score = 100
        elif is_passed:
            status = '양호'
            score = 70
        elif value < criterion['min']:
            status = '낮음'
            score = 40
        else:
            status = '높음'
            score = 40

        if is_passed:
            evaluation['passed_checks'] += 1

        evaluation['details'][key] = {
            'name': criterion['name'],
            'value': round(value, 2),
            'unit': criterion['unit'],
            'range': f"{criterion['min']}-{criterion['max']}",
            'optimal_range': f"{criterion['optimal'][0]}-{criterion['optimal'][1]}",
            'status': status,
            'passed': is_passed,
            'score': score,
            'description': criterion['description']
        }

    # 전체 점수 계산
    total_score = sum(detail['score'] for detail in evaluation['details'].values())
    evaluation['overall_score'] = round(total_score / len(criteria), 1)

    # 전체 평가
    if evaluation['overall_score'] >= 85:
        evaluation['overall_grade'] = 'A (우수)'
        evaluation['summary'] = '매우 좋은 강의 품질입니다.'
    elif evaluation['overall_score'] >= 70:
        evaluation['overall_grade'] = 'B (양호)'
        evaluation['summary'] = '전반적으로 양호한 품질입니다.'
    elif evaluation['overall_score'] >= 60:
        evaluation['overall_grade'] = 'C (보통)'
        evaluation['summary'] = '일부 개선이 필요합니다.'
    else:
        evaluation['overall_grade'] = 'D (개선필요)'
        evaluation['summary'] = '여러 측면에서 개선이 필요합니다.'

    # 개선 제안
    evaluation['recommendations'] = []

    for key, detail in evaluation['details'].items():
        if not detail['passed']:
            if key == 'rms_energy':
                if detail['value'] < criteria[key]['min']:
                    evaluation['recommendations'].append('🔊 볼륨을 높여주세요 (마이크 게인 조정)')
                else:
                    evaluation['recommendations'].append('🔉 볼륨을 낮춰주세요 (과포화 방지)')

            elif key == 'spectral_centroid':
                if detail['value'] < criteria[key]['min']:
                    evaluation['recommendations'].append('🎤 마이크를 가까이하고 또렷하게 발음해주세요')
                else:
                    evaluation['recommendations'].append('🎤 마이크 거리를 적절히 조정해주세요')

            elif key == 'spectral_rolloff':
                if detail['value'] < criteria[key]['min']:
                    evaluation['recommendations'].append('📻 더 좋은 마이크나 녹음 환경을 사용해주세요')

            elif key == 'tempo':
                if detail['value'] < criteria[key]['min']:
                    evaluation['recommendations'].append('⏩ 조금 더 빠르게 말씀해주세요')
                else:
                    evaluation['recommendations'].append('⏸️ 조금 더 천천히 말씀해주세요')

            elif key == 'silence_ratio':
                if detail['value'] < criteria[key]['min']:
                    evaluation['recommendations'].append('🫁 적절한 쉬는 시간을 가져주세요')
                else:
                    evaluation['recommendations'].append('✂️ 긴 침묵 구간을 편집해주세요')

            elif key == 'mfcc_stability':
                if detail['value'] > criteria[key]['max']:
                    evaluation['recommendations'].append('🔇 배경 소음을 줄여주세요')

    if not evaluation['recommendations']:
        evaluation['recommendations'].append('✅ 모든 항목이 적절합니다!')

    return evaluation
