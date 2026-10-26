# -*- coding: utf-8 -*-
"""
演示：一段音频是怎么变成「特征矩阵」的
（这里用 librosa 的梅尔谱做直观例子；项目里实际用的是 wav2vec2，思路完全一样）
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import librosa

# ============ 第 1 步：造一段 1 秒的音频 ============
sr = 16000                                      # 采样率：每秒记录 16000 个数字
t = np.linspace(0, 1, sr, endpoint=False)       # 0 到 1 秒之间的 16000 个时间点
wave = 0.3 * np.sin(2 * np.pi * 440 * t)        # 440Hz 正弦波（标准音 A），0.3 是音量

print('=== 第 1 步：原始音频其实就是「一串数字」 ===')
print('音频的采样点数:', len(wave), '(1秒 x 16000Hz)')
print('前 10 个采样点的数值:', np.round(wave[:10], 4))
print()

# ============ 第 2 步：提取梅尔谱，把音频变成矩阵 ============
# melspectrogram 计算梅尔谱；power_to_db 把能量转成分贝
mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=512, hop_length=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

print('=== 第 2 步：提取梅尔谱，变成「矩阵」 ===')
print('矩阵形状 (频率方向 x 时间帧):', mel.shape)
print('矩阵前 5 行 x 5 列（单位是分贝）:')
print(np.round(mel_db[:5, :5], 2))
