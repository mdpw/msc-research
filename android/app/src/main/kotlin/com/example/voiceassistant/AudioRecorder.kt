package com.example.voiceassistant

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat

class AudioRecorder(private val context: Context) {

    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val TAG = "AudioRecorder"

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val audioBuffer = mutableListOf<Float>()
    private var recordingThread: Thread? = null

    fun startRecording() {
        if (ActivityCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e(TAG, "❌ Audio permission not granted")
            throw SecurityException("Audio permission not granted")
        }

        try {
            val bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT
            ) * 2

            Log.d(TAG, "📱 Creating AudioRecord with buffer size: $bufferSize")

            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "❌ AudioRecord not initialized properly")
                throw IllegalStateException("AudioRecord initialization failed")
            }

            audioBuffer.clear()
            audioRecord?.startRecording()
            isRecording = true

            Log.d(TAG, "🎙️ Recording started")

            recordingThread = Thread {
                val buffer = ShortArray(bufferSize / 2)

                while (isRecording) {
                    val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0

                    if (read > 0) {
                        // Convert to float and store
                        for (i in 0 until read) {
                            audioBuffer.add(buffer[i] / 32768.0f)
                        }
                    } else {
                        Log.e(TAG, "⚠️ Read returned: $read")
                    }
                }
                Log.d(TAG, "📊 Recording thread finished")
            }
            recordingThread?.start()

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error starting recording: ${e.message}", e)
            throw e
        }
    }

    fun stopRecording(): FloatArray {
        try {
            Log.d(TAG, "⏹️ Stopping recording...")
            isRecording = false

            // Wait for recording thread to finish
            recordingThread?.join(1000)

            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            val result = audioBuffer.toFloatArray()
            audioBuffer.clear()

            Log.d(TAG, "✅ Recording stopped - Captured ${result.size} samples (${result.size / SAMPLE_RATE.toFloat()} seconds)")

            return result

        } catch (e: Exception) {
            Log.e(TAG, "❌ Error stopping recording: ${e.message}", e)
            return FloatArray(0)
        }
    }
}